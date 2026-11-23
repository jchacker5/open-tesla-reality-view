"""Shutdown, SIGTERM and orphaned builds: a studio that stops must not leave a build running behind it."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from tsplat.studio import jobs
from tsplat.studio.jobs import Phase, Runner

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the job runner uses POSIX process groups")

# A fake long-running build step: writes its pid, then sleeps. With a third argument it ignores SIGTERM.
SLEEPER = textwrap.dedent(
    """
    import os, signal, sys, time
    if len(sys.argv) > 2:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    open(sys.argv[1], "w").write(str(os.getpid()))
    print("$ colmap feature_extractor ...", flush=True)
    time.sleep(60)
    """
)


@pytest.fixture
def sleeper(tmp_path):
    p = tmp_path / "sleeper.py"
    p.write_text(SLEEPER)
    return str(p)


def wait_pid(p, timeout=10) -> int:
    end = time.time() + timeout
    while time.time() < end:
        if p.exists() and p.read_text():
            return int(p.read_text())
        time.sleep(0.05)
    raise AssertionError(f"{p} never appeared")


def gone(pgid, timeout=10) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if not jobs.pgid_alive(pgid):
            return True
        time.sleep(0.05)
    return False


def wait_state(r: Runner, name: str, states, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if r.get(name).state in states:
            return r.get(name)
        time.sleep(0.05)
    raise AssertionError(f"{name} stuck in {r.get(name).state}")


def runner(tmp_path, argv) -> Runner:
    r = Runner(tmp_path / "work", plan=lambda job: [Phase("x", argv, ("features",))], progress_interval=0)
    r.start()
    return r


def saved(tmp_path, name):
    return json.loads((tmp_path / "work" / name / "job.json").read_text())


def test_running_job_records_its_process_group(tmp_path, sleeper):
    pidf = tmp_path / "pid"
    r = runner(tmp_path, [sys.executable, sleeper, str(pidf)])
    try:
        r.submit("slow", {})
        pid = wait_pid(pidf)
        assert saved(tmp_path, "slow")["state"] == "running" and saved(tmp_path, "slow")["pgid"] == pid
    finally:
        r.stop()
    assert saved(tmp_path, "slow")["pgid"] is None


@pytest.mark.parametrize("stubborn", [False, True])
def test_stop_kills_the_build_and_marks_it_interrupted(tmp_path, sleeper, monkeypatch, stubborn):
    monkeypatch.setattr(jobs, "KILL_GRACE", 0.5)
    pidf = tmp_path / "pid"
    r = runner(tmp_path, [sys.executable, sleeper, str(pidf)] + (["stubborn"] if stubborn else []))
    r.submit("slow", {})
    pid = wait_pid(pidf)
    time.sleep(0.2)
    t0 = time.time()
    r.stop()  # blocks: SIGTERM, then SIGKILL after KILL_GRACE
    assert time.time() - t0 < 5
    assert not jobs.pgid_alive(pid)
    job = r.get("slow")
    assert job.state == "interrupted" and "resume" in job.error
    assert saved(tmp_path, "slow")["state"] == "interrupted"
    r.stop()  # idempotent


def test_external_sigterm_is_interrupted_not_failed(tmp_path, sleeper):
    pidf = tmp_path / "pid"
    r = runner(tmp_path, [sys.executable, sleeper, str(pidf)])
    try:
        r.submit("slow", {})
        os.kill(wait_pid(pidf), signal.SIGTERM)
        job = wait_state(r, "slow", jobs.TERMINAL)
        assert job.state == "interrupted", job.error
        assert "SIGTERM" in job.error
        assert r.resume("slow").state == "queued"
        r.cancel("slow")
    finally:
        r.stop()


def make_orphan(tmp_path, sleeper) -> subprocess.Popen:
    """A job.json that says running, with a live process group a dead studio left behind."""
    pidf = tmp_path / "orphan.pid"
    proc = subprocess.Popen([sys.executable, sleeper, str(pidf)], start_new_session=True)
    wait_pid(pidf)
    d = tmp_path / "work" / "old"
    d.mkdir(parents=True)
    (d / "job.json").write_text(
        json.dumps({"name": "old", "state": "running", "spec": {}, "created": 1.0, "pgid": proc.pid})
    )
    return proc


def test_orphaned_running_job_refuses_resume_until_killed(tmp_path, sleeper, monkeypatch):
    monkeypatch.setattr(jobs, "KILL_GRACE", 0.5)
    proc = make_orphan(tmp_path, sleeper)
    try:
        r = Runner(tmp_path / "work", plan=lambda job: [])
        job = r.get("old")
        assert job.state == jobs.ORPHANED and str(proc.pid) in job.error
        assert saved(tmp_path, "old")["state"] == jobs.ORPHANED
        assert r.busy("old")
        with pytest.raises(ValueError, match="kill it first"):
            r.resume("old")
        with pytest.raises(ValueError):
            r.submit("old", {})
        with pytest.raises(ValueError):
            r.forget("old")
        r.kill_orphan("old", wait=True)
        proc.wait(5)
        assert gone(proc.pid)
        assert r.get("old").state == "interrupted"
        assert saved(tmp_path, "old")["state"] == "interrupted"
        assert r.resume("old").state == "queued"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_orphan_that_exits_on_its_own_becomes_interrupted(tmp_path, sleeper):
    proc = make_orphan(tmp_path, sleeper)
    r = Runner(tmp_path / "work", plan=lambda job: [])
    assert r.get("old").state == jobs.ORPHANED
    proc.kill()
    proc.wait(5)
    assert gone(proc.pid)
    assert r.get("old").state == "interrupted"
    assert [j.state for j in r.list()] == ["interrupted"]


def test_cancel_kills_an_orphan(tmp_path, sleeper, monkeypatch):
    monkeypatch.setattr(jobs, "KILL_GRACE", 0.5)
    proc = make_orphan(tmp_path, sleeper)
    try:
        r = Runner(tmp_path / "work", plan=lambda job: [])
        r.cancel("old")
        proc.wait(5)
        assert wait_state(r, "old", ("interrupted",)).state == "interrupted"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_dead_pgid_loads_as_interrupted(tmp_path):
    d = tmp_path / "work" / "old"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"name": "old", "state": "running", "spec": {}, "pgid": 2**22 + 7}))
    job = Runner(tmp_path / "work", plan=lambda job: []).get("old")
    assert job.state == "interrupted" and job.pgid is None


# ---- the HTTP server: SSE, the kill endpoint, and a real process getting ctrl-c / SIGTERM / SIGHUP ---------


def studio_client(tmp_path, r: Runner):
    pytest.importorskip("fastapi", reason="needs `uv sync --extra studio`")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from tsplat.studio import footage, server

    cfg = server.Config(tmp_path / "data", tmp_path / "work", 8799, token="tok", cwd=tmp_path)
    app = server.create_app(cfg, library=footage.Library(cfg.data, autodetect_fn=lambda: []), runner=r)
    return TestClient(app, base_url="http://127.0.0.1:8799", headers={"X-OTRV-Token": "tok"}), cfg


def test_sse_for_unknown_job_is_404(tmp_path):
    c, _ = studio_client(tmp_path, Runner(tmp_path / "work", plan=lambda job: []))
    assert c.get("/api/jobs/nope/events").status_code == 404


def test_sse_ends_when_the_server_stops(tmp_path, sleeper):
    import threading

    pidf = tmp_path / "pid"
    r = Runner(tmp_path / "work", plan=lambda job: [Phase("x", [sys.executable, sleeper, str(pidf)], ("features",))])
    c, cfg = studio_client(tmp_path, r)
    with c:
        r.submit("slow", {})
        wait_pid(pidf)
        threading.Timer(0.5, cfg.stopping.set).start()
        t0 = time.time()
        with c.stream("GET", "/api/jobs/slow/events") as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert time.time() - t0 < 3  # ended by the stopping flag, not by the job (which sleeps 60 s)
        assert "colmap feature_extractor" in body
    assert r.get("slow").state == "interrupted"  # lifespan shutdown stopped the runner


def test_kill_endpoint(tmp_path, sleeper, monkeypatch):
    monkeypatch.setattr(jobs, "KILL_GRACE", 0.5)
    proc = make_orphan(tmp_path, sleeper)
    try:
        c, _ = studio_client(tmp_path, Runner(tmp_path / "work", plan=lambda job: []))
        assert c.get("/api/jobs/old").json()["state"] == jobs.ORPHANED
        assert c.post("/api/jobs/old/resume").status_code == 409
        assert c.post("/api/jobs/old/kill").status_code == 200
        proc.wait(5)
        end = time.time() + 5
        while c.get("/api/jobs/old").json()["state"] != "interrupted" and time.time() < end:
            time.sleep(0.05)
        assert c.get("/api/jobs/old").json()["state"] == "interrupted"
        assert c.post("/api/jobs/old/kill").status_code == 409  # nothing left to kill
    finally:
        if proc.poll() is None:
            proc.kill()


SERVE = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from tsplat.studio import footage, jobs, server

    tmp, port, sleeper = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    pidf = tmp / "pid"
    runner = jobs.Runner(
        tmp / "work", plan=lambda job: [jobs.Phase("x", [sys.executable, sleeper, str(pidf)], ("features",))]
    )
    cfg = server.Config(tmp / "data", tmp / "work", port, token="tok", cwd=tmp, city_port=server.free_port(port + 1))
    app = server.create_app(cfg, library=footage.Library(cfg.data, autodetect_fn=lambda: []), runner=runner)
    runner.submit("slow", {})
    server.run_servers(cfg, [(app, port), (server.create_city_app(cfg), cfg.city_port)], runner)
    print("clean exit", flush=True)
    """
)


@pytest.mark.parametrize("sig", ["SIGINT", "SIGTERM", "SIGHUP"])
def test_signal_stops_the_server_and_kills_the_build(tmp_path, sleeper, sig):
    """A real `otrv studio`-style process with a running build and an open SSE stream: one signal stops
    it within seconds, the build's process group dies with it, and the job is resumable (interrupted)."""
    pytest.importorskip("fastapi", reason="needs `uv sync --extra studio`")
    import threading
    import urllib.request

    from tsplat.studio import server

    port = server.free_port(8850)
    script = tmp_path / "serve.py"
    script.write_text(SERVE)
    proc = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), str(port), sleeper],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        pid = wait_pid(tmp_path / "pid", timeout=20)
        assert jobs.pgid_alive(pid)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/jobs/slow/events", headers={"X-OTRV-Token": "tok"})
        stream = urllib.request.urlopen(req, timeout=20)  # held open while we signal
        reader = threading.Thread(target=stream.read, daemon=True)
        reader.start()
        time.sleep(0.3)
        t0 = time.time()
        proc.send_signal(getattr(signal, sig))
        out, _ = proc.communicate(timeout=15)
        assert time.time() - t0 < server.GRACEFUL_SHUTDOWN - 0.5, out  # the SSE stream ended at once
        assert "clean exit" in out, out
        assert gone(pid, timeout=2)
        job = json.loads((tmp_path / "work" / "slow" / "job.json").read_text())
        assert job["state"] == "interrupted" and job["pgid"] is None
    finally:
        if proc.poll() is None:
            proc.kill()
        if jobs.pgid_alive(_read_pid(tmp_path / "pid")):
            os.killpg(_read_pid(tmp_path / "pid"), signal.SIGKILL)


def _read_pid(p) -> int | None:
    try:
        return int(p.read_text())
    except (OSError, ValueError):
        return None

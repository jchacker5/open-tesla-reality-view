import json
import sys
import textwrap
import time

import pytest

from tsplat.studio import jobs
from tsplat.studio.jobs import Job, Phase, Runner, read_events

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the job runner uses a POSIX PTY")

PREP = textwrap.dedent(
    """
    import sys
    print("[t] 4 cams, 0s +2s @ 2 fps", flush=True)
    print("ego mask: guessed from the window", flush=True)
    print("pairs: 10", flush=True)
    """
)
RECON = textwrap.dedent(
    """
    import sys, time, pathlib
    flag = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    print("$ colmap feature_extractor ...", flush=True)
    for i in range(1, 9):
        print(f"I2026 f.cc:1] Processed file [{i}/8]", flush=True)
    print("$ colmap matches_importer ...", flush=True)
    print("Processing block [1/1]", flush=True)
    print("$ colmap mapper ...", flush=True)
    print("Registering image #1 (num_reg_frames=4)", flush=True)
    print("$ colmap image_undistorter ...", flush=True)
    for p in range(2):
        for i in range(1, 9):
            print(f"Undistorting image [{i}/8]", flush=True)
    if flag is not None and not flag.exists():
        flag.write_text("x")
        print("error: brush crashed", flush=True)
        sys.exit(3)
    print("$ /opt/brush_app dense", flush=True)
    for i in range(0, 101, 20):
        sys.stdout.write(f"\\r\\x1b[2K\\x1b[32m{i}/100 Steps (1000.0/s, 0s remaining)\\x1b[0m")
        sys.stdout.flush()
        time.sleep(0.01)
    print("\\nsplat -> scene.ply (1 MB)", flush=True)
    """
)
SLEEPER = textwrap.dedent(
    """
    import signal, sys, time
    if len(sys.argv) > 1:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # stubborn: needs the SIGKILL
    print("$ colmap feature_extractor ...", flush=True)
    print("Processed file [1/8]", flush=True)
    time.sleep(60)
    """
)


@pytest.fixture
def scripts(tmp_path):
    out = {}
    for name, body in {"prep": PREP, "recon": RECON, "sleeper": SLEEPER}.items():
        p = tmp_path / f"{name}.py"
        p.write_text(body)
        out[name] = str(p)
    return out


def wait(runner: Runner, name: str, states=jobs.TERMINAL, timeout: float = 20) -> Job:
    end = time.time() + timeout
    while time.time() < end:
        j = runner.get(name)
        if j and j.state in states:
            return j
        time.sleep(0.05)
    raise AssertionError(f"{name} stuck in {runner.get(name).state}")


def make_runner(tmp_path, plan) -> Runner:
    r = Runner(tmp_path / "work", plan=plan, units=lambda job: (8, 100, 4), progress_interval=0)
    r.start()
    return r


def two_phase(scripts, recon_args=()):
    py = sys.executable
    return lambda job: [
        Phase("prep", [py, scripts["prep"]], ("frames", "masks")),
        Phase("recon", [py, scripts["recon"], *recon_args], ("features", "match", "map", "undistort", "train")),
        Phase("finalize", lambda job, log: log("city ready"), ("export",)),
    ]


def test_job_runs_and_streams_progress(tmp_path, scripts):
    r = make_runner(tmp_path, two_phase(scripts))
    try:
        r.submit("city1", {"x": 1})
        job = wait(r, "city1")
        assert job.state == "done", job.error
        evs = read_events(r.events_path("city1"))
        ids = [e["id"] for e in evs]
        assert ids == sorted(set(ids)) and ids[0] == 1  # unique, increasing
        states = [e["state"] for e in evs if e["type"] == "state"]
        assert states == ["queued", "running", "done"]
        starts = [e["stage"] for e in evs if e["type"] == "stage" and e["status"] == "start"]
        assert starts == list(jobs.progress.STAGES)
        train = [e for e in evs if e["type"] == "progress" and e["stage"] == "train"]
        assert train[-1]["cur"] == 100 and train[-1]["frac"] == 1.0
        logs = [e["line"] for e in evs if e["type"] == "log"]
        assert "pairs: 10" in logs and "city ready" in logs
        assert not any("Steps (" in ln for ln in logs)  # progress redraws don't flood the log
        stages = r.work / "city1" / "stages"
        assert sorted(p.stem for p in stages.glob("*.done")) == sorted(jobs.progress.STAGES)
        assert json.loads((stages / "features.done").read_text())["units"] == 8
        saved = json.loads((r.work / "city1" / "job.json").read_text())
        assert saved["state"] == "done" and saved["spec"] == {"x": 1}
        assert "\x1b[" in (r.work / "city1" / "studio.log").read_text()  # raw output kept (it ran under a PTY)
    finally:
        r.stop()


def test_failure_then_resume_skips_finished_stages(tmp_path, scripts):
    flag = tmp_path / "crashed-once"
    r = make_runner(tmp_path, two_phase(scripts, [str(flag)]))
    try:
        r.submit("c", {})
        job = wait(r, "c")
        assert job.state == "failed"
        assert "recon exited with code 3" in job.error and "brush crashed" in job.error
        assert job.to_dict()["stages_done"][:2] == ["frames", "masks"]
        assert "train" not in job.to_dict()["stages_done"]
        last = read_events(r.events_path("c"))[-1]["id"]
        r.resume("c")
        job = wait(r, "c", ("done", "failed"))
        assert job.state == "done", job.error
        evs = read_events(r.events_path("c"))
        assert all(e["id"] > last for e in evs if e["id"] > last) and len({e["id"] for e in evs}) == len(evs)
        after = [e for e in evs if e["id"] > last]
        phases = [e["phase"] for e in after if e["type"] == "phase"]
        assert phases == ["recon", "finalize"]  # prep was not run again; recon (one command) reran whole
        assert job.attempts == 2
    finally:
        r.stop()


def test_cancel_running_job_kills_process_group(tmp_path, scripts):
    r = make_runner(tmp_path, lambda job: [Phase("recon", [sys.executable, scripts["sleeper"]], ("features",))])
    try:
        r.submit("slow", {})
        wait(r, "slow", ("running",))
        deadline = time.time() + 10
        while time.time() < deadline and not r.get("slow").frac.get("features"):
            time.sleep(0.05)
        t0 = time.time()
        r.cancel("slow")
        job = wait(r, "slow")
        assert job.state == "cancelled"
        assert time.time() - t0 < 4  # SIGTERM was enough
        assert r.resume("slow").state == "queued"
        r.cancel("slow")
    finally:
        r.stop()


def test_cancel_escalates_to_sigkill(tmp_path, scripts, monkeypatch):
    monkeypatch.setattr(jobs, "KILL_GRACE", 0.5)
    r = make_runner(tmp_path, lambda job: [Phase("x", [sys.executable, scripts["sleeper"], "stubborn"], ("features",))])
    try:
        r.submit("stubborn", {})
        wait(r, "stubborn", ("running",))
        time.sleep(0.5)  # let it install the SIGTERM handler
        r.cancel("stubborn")
        assert wait(r, "stubborn", timeout=10).state == "cancelled"
    finally:
        r.stop()


def test_fifo_queue_and_cancel_queued(tmp_path, scripts):
    r = make_runner(tmp_path, lambda job: [Phase("x", [sys.executable, scripts["sleeper"]], ("features",))])
    try:
        r.submit("first", {})
        r.submit("second", {})
        r.submit("third", {})
        wait(r, "first", ("running",))
        assert r.get("second").state == "queued"
        r.cancel("second")
        assert r.get("second").state == "cancelled"
        r.cancel("first")
        wait(r, "third", ("running",))  # skipped the cancelled one
        assert r.get("second").state == "cancelled"
        with pytest.raises(ValueError):
            r.submit("third", {})  # already running
    finally:
        r.stop()


def test_missing_executable_fails_cleanly(tmp_path):
    r = make_runner(tmp_path, lambda job: [Phase("x", ["/nonexistent/otrv"], ("frames",))])
    try:
        r.submit("bad", {})
        job = wait(r, "bad")
        assert job.state == "failed" and "could not start" in job.error
    finally:
        r.stop()


def test_optional_phase_failure_is_not_fatal(tmp_path, scripts):
    def boom(job, log):
        raise RuntimeError("no metrics")

    r = make_runner(tmp_path, lambda job: [Phase("bench", boom, ("export",), required=False)])
    try:
        r.submit("opt", {})
        assert wait(r, "opt").state == "done"
    finally:
        r.stop()


def test_running_job_becomes_interrupted_after_restart(tmp_path):
    d = tmp_path / "work" / "old"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"name": "old", "state": "running", "spec": {}, "created": 1.0}))
    r = Runner(tmp_path / "work", plan=lambda job: [])
    job = r.get("old")
    assert job.state == "interrupted" and "resume" in job.error
    assert json.loads((d / "job.json").read_text())["state"] == "interrupted"


def test_pipeline_v2_substages_reach_the_event_log(tmp_path):
    script = tmp_path / "v2.py"
    script.write_text(
        "print('[t] 6 cams, 0s +2s @ 6 fps')\nprint('ego mask: guessed')\nprint('sky (segformer): front 30%')\n"
        "print('relight: something pipeline v3 does')\n",
    )
    r = make_runner(tmp_path, lambda job: [Phase("prep", [sys.executable, str(script)], ("frames", "masks"))])
    try:
        r.submit("v2", {})
        assert wait(r, "v2").state == "done"
        evs = read_events(r.events_path("v2"))
        subs = [e for e in evs if e["type"] == "substage"]
        assert [(e["stage"], e["substage"], e["label"]) for e in subs] == [("masks", "sky", "Sky masks")]
        assert any(e["type"] == "log" and e["line"].startswith("relight:") for e in evs)  # unknown: just logged
    finally:
        r.stop()


@pytest.mark.parametrize("name", ["", "../x", "a/b", ".hidden", "x" * 81, "a..b"])
def test_rejects_bad_names(tmp_path, name):
    r = Runner(tmp_path / "work", plan=lambda job: [])
    with pytest.raises(ValueError):
        r.submit(name, {})

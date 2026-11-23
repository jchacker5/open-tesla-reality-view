import json
import shutil
import sys
import time

import pytest

pytest.importorskip("fastapi", reason="needs `uv sync --extra studio`")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from tsplat.studio import footage, jobs, server, synth  # noqa: E402

PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"
TOKEN = "t0ken-for-tests"
HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg/ffprobe for synthetic clips")


def make_app(tmp_path, runner=None, autodetect=lambda: []):
    cfg = server.Config(tmp_path / "data", tmp_path / "work", PORT, token=TOKEN, cwd=tmp_path)
    lib = footage.Library(cfg.data, autodetect_fn=autodetect)
    return server.create_app(cfg, library=lib, runner=runner), cfg, lib


@pytest.fixture
def client(tmp_path):
    app, cfg, lib = make_app(tmp_path)
    return TestClient(app, base_url=BASE)


AUTH = {"X-OTRV-Token": TOKEN}


# ---- token, Host and Origin ----------------------------------------------------------------------------


def test_api_requires_token(client):
    assert client.get("/api/sources").status_code == 401
    assert client.get("/api/sources", headers={"X-OTRV-Token": "nope"}).status_code == 401
    assert client.get("/api/sources", headers=AUTH).status_code == 200
    assert client.get(f"/api/sources?token={TOKEN}").status_code == 200  # <img>/<video>/EventSource fallback


def test_index_swaps_token_for_cookie(client):
    assert client.get("/").status_code == 401
    assert client.get("/?token=wrong").status_code == 401
    r = client.get(f"/?token={TOKEN}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"]
    assert "httponly" in cookie.lower() and "samesite=strict" in cookie.lower()
    r = client.get("/")  # the client kept the cookie
    assert r.status_code == 200
    assert f'<meta name="otrv-token" content="{TOKEN}">' in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY"
    assert client.get("/api/sources").status_code == 200  # GET with the cookie alone is fine


def test_state_changes_need_the_header_not_just_the_cookie(client):
    client.get(f"/?token={TOKEN}")
    assert client.post("/api/sources/rescan").status_code == 401  # cookie only: CSRF-proof
    assert client.post(f"/api/sources/rescan?token={TOKEN}").status_code == 401
    assert client.post("/api/sources/rescan", headers=AUTH).status_code == 200


@pytest.mark.parametrize("host", ["evil.example:8799", "127.0.0.1:9999", "localhost", "0.0.0.0:8799"])
def test_rejects_foreign_host(client, host):
    r = client.get("/api/sources", headers={**AUTH, "Host": host})
    assert r.status_code == 403 and "Host" in r.json()["detail"]


def test_accepts_localhost_host(client):
    assert client.get("/api/sources", headers={**AUTH, "Host": f"localhost:{PORT}"}).status_code == 200


@pytest.mark.parametrize("origin", ["https://evil.example", "http://127.0.0.1:1234", "null"])
def test_rejects_foreign_origin(client, origin):
    assert client.post("/api/sources/rescan", headers={**AUTH, "Origin": origin}).status_code == 403
    assert client.get("/api/sources", headers={**AUTH, "Origin": origin}).status_code == 403


def test_same_origin_and_sec_fetch(client):
    assert client.post("/api/sources/rescan", headers={**AUTH, "Origin": BASE}).status_code == 200
    r = client.get("/api/sources", headers={**AUTH, "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_static_and_security_headers(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert client.get("/api/sources", headers=AUTH).headers["cache-control"] == "no-store"


def test_frontend_makes_no_external_requests():
    """Nothing leaves the machine: no http(s) URLs in the studio's own static files."""
    import re

    for f in server.STATIC.iterdir():
        text = f.read_text()
        urls = [u for u in re.findall(r"https?://[^\s\"')]+", text) if "www.w3.org/2000/svg" not in u]
        assert urls == [], f"{f.name} references {urls}"


def test_cities_need_token(tmp_path):
    app, cfg, _ = make_app(tmp_path)
    c = TestClient(app, base_url=BASE)
    city = cfg.work / "drive1" / "viewer"
    city.mkdir(parents=True)
    (city / "scene.ply").write_bytes(b"ply\n" + b"\0" * 100)
    (city / "path.json").write_text("[]")
    assert c.get("/cities/drive1/").status_code == 401
    c.get(f"/?token={TOKEN}")
    r = c.get("/cities/drive1/")
    assert r.status_code == 200 and "<html" in r.text.lower()
    assert "default-src" not in r.headers["content-security-policy"]  # viewer still loads three/Spark from a CDN
    assert c.get("/cities/drive1/path.json").json() == []
    r = c.get("/cities/drive1/scene.ply", headers={"Range": "bytes=0-3"})
    assert r.status_code == 206 and r.content == b"ply\n"
    assert c.get("/cities/drive1/../../secret").status_code == 404
    assert c.get("/cities/nope/").status_code == 404
    r = c.get("/cities/drive1?mode=drive", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/cities/drive1/?mode=drive"


def test_gallery_and_delete(tmp_path):
    app, cfg, _ = make_app(tmp_path)
    c = TestClient(app, base_url=BASE)
    d = cfg.work / "drive1"
    (d / "viewer").mkdir(parents=True)
    (d / "viewer" / "scene.ply").write_bytes(b"x" * 1000)
    (d / "viewer" / "path.json").write_text("[]")
    (d / "meta.json").write_text(
        json.dumps({"clip": "c", "timestamp": "2026-01-02_03-04-05", "start": 3, "dur": 20, "cams": ["front"]})
    )
    (d / "metrics.json").write_text(json.dumps({"psnr": {"overall": 23.4}, "gate_failures": []}))
    (cfg.work / "not-a-city").mkdir()
    cities = c.get("/api/cities", headers=AUTH).json()["cities"]
    assert [x["name"] for x in cities] == ["drive1"]
    assert cities[0]["psnr"] == 23.4 and cities[0]["splat_bytes"] == 1000 and cities[0]["dur"] == 20
    assert c.delete("/api/cities/../etc", headers=AUTH).status_code == 404
    assert c.delete("/api/cities/drive1").status_code == 401  # needs the header
    assert c.delete("/api/cities/drive1", headers=AUTH).json() == {"deleted": "drive1"}
    assert not d.exists()


def test_add_source_validation(client, tmp_path):
    r = client.post("/api/sources", json={"path": str(tmp_path / "missing")}, headers=AUTH)
    assert r.status_code == 400 and "not a folder" in r.json()["detail"]
    empty = tmp_path / "empty"
    empty.mkdir()
    r = client.post("/api/sources", json={"path": str(empty)}, headers=AUTH)
    assert r.status_code == 400 and "no TeslaCam clips" in r.json()["detail"]


def test_upload_rejects_non_clips(client):
    r = client.put("/api/uploads/RecentClips/notes.txt", content=b"x", headers=AUTH)
    assert r.status_code == 400
    r = client.put("/api/uploads/../2026-01-02_03-04-05-front.mp4", content=b"x", headers=AUTH)
    assert r.status_code in (400, 404)


# ---- the API on a synthetic TeslaCam tree --------------------------------------------------------------


@pytest.fixture(scope="module")
def teslacam(tmp_path_factory):
    if not HAS_FFMPEG:
        pytest.skip("needs ffmpeg")
    root = tmp_path_factory.mktemp("usb") / "TeslaCam"
    synth.make_teslacam(root, seconds=6, fps=10, size="160x120", small=True)
    return root


FAKE_PIPELINE = """
import sys, time
print("[x] 4 cams, 0s +5s @ 1 fps", flush=True)
print("ego mask: guessed", flush=True)
print("$ colmap feature_extractor ...", flush=True)
for i in range(1, 5):
    print(f"Processed file [{i}/4]", flush=True)
    time.sleep(0.02)
print("$ /x/brush_app dense", flush=True)
print("100/100 Steps (100.0/s, 0s remaining)", flush=True)
print("splat -> scene.ply", flush=True)
"""


@needs_ffmpeg
def test_drives_clips_and_jobs(tmp_path, teslacam):
    script = tmp_path / "fake.py"
    script.write_text(FAKE_PIPELINE)
    runner = jobs.Runner(
        tmp_path / "work",
        plan=lambda job: [jobs.Phase("all", [sys.executable, str(script)], tuple(jobs.progress.STAGES))],
        units=lambda job: (4, 100, 1),
        progress_interval=0,
    )
    app, cfg, lib = make_app(tmp_path, runner=runner, autodetect=lambda: [teslacam])
    with TestClient(app, base_url=BASE) as c:  # lifespan: starts the worker and the scan
        lib.scan(wait=True)
        src = c.get("/api/sources", headers=AUTH).json()
        assert src["sources"][0]["kind"] == "usb" and src["sources"][0]["clips"] == 2

        drives = c.get("/api/drives", headers=AUTH).json()["drives"]
        assert len(drives) == 1
        d = drives[0]
        assert d["n_clips"] == 2 and d["n_cams"] == 4 and d["seconds"] == pytest.approx(12, abs=0.5)
        assert d["mean_mph"] and 5 < d["mean_mph"] < 40
        assert len(d["route"]) >= 2 and len(d["speed"]) >= 10
        assert d["thumb_cams"] == ["front", "back", "left_repeater", "right_repeater"]

        full = c.get(f"/api/drives/{d['id']}", headers=AUTH).json()
        assert [x["offset"] for x in full["clips"]] == [0, 6]
        sug = c.get(f"/api/drives/{d['id']}/suggest?dur=5", headers=AUTH).json()["suggestions"]
        assert sug and sug[0]["clip"] == full["clips"][1]["key"] and sug[0]["offset"] >= 6
        assert c.get("/api/drives/nope", headers=AUTH).status_code == 404

        key = full["clips"][1]["key"]
        th = c.get(f"/api/clips/{key}/thumb.jpg?cam=back", headers=AUTH)
        assert th.status_code == 200 and th.content[:2] == b"\xff\xd8"
        assert c.get(f"/api/clips/{key}/thumb.jpg?cam=left_pillar", headers=AUTH).status_code == 404
        v = c.get(f"/api/clips/{key}/video/front", headers={**AUTH, "Range": "bytes=0-99"})
        assert v.status_code == 206 and len(v.content) == 100 and v.headers["content-type"] == "video/mp4"

        est = c.get("/api/estimate?cams=4&dur=20&fps=3&steps=15000", headers=AUTH).json()
        assert est["images"] == 240 and est["total"] > 0 and set(est["stages"]) == set(jobs.progress.STAGES)

        bad = c.post("/api/jobs", json={"clip": key, "start": 4, "dur": 5}, headers=AUTH)
        assert bad.status_code == 400 and "past the end" in bad.json()["detail"]
        r = c.post("/api/jobs", json={"clip": key, "start": 0, "dur": 5, "fps": 1, "steps": 100}, headers=AUTH)
        assert r.status_code == 200, r.text
        name = r.json()["name"]
        assert name == "drive_2026-01-02_03-04-11"
        assert r.json()["spec"]["clip"]["cams"]["front"].endswith("-front.mp4")

        end = time.time() + 20
        while c.get(f"/api/jobs/{name}", headers=AUTH).json()["state"] != "done" and time.time() < end:
            time.sleep(0.1)
        assert c.get(f"/api/jobs/{name}", headers=AUTH).json()["state"] == "done"
        assert [j["name"] for j in c.get("/api/jobs", headers=AUTH).json()["jobs"]] == [name]

        # SSE: full replay, then replay from Last-Event-ID
        body = c.get(f"/api/jobs/{name}/events", headers=AUTH).text
        evs = [json.loads(ln[6:]) for ln in body.splitlines() if ln.startswith("data: {") and ln != "data: {}"]
        assert evs[0]["type"] == "state" and evs[-1] == {**evs[-1], "type": "state", "state": "done"}
        assert any(e["type"] == "progress" and e["stage"] == "features" for e in evs)
        assert "event: end" in body
        mid = evs[len(evs) // 2]["id"]
        body2 = c.get(f"/api/jobs/{name}/events", headers={**AUTH, "Last-Event-ID": str(mid)}).text
        ids2 = [json.loads(ln[6:])["id"] for ln in body2.splitlines() if ln.startswith("data: {") and ln != "data: {}"]
        assert ids2 and min(ids2) == mid + 1
        assert f"id: {mid + 1}" in body2

        assert c.post(f"/api/jobs/{name}/resume", headers=AUTH).status_code == 409  # done jobs don't resume
        assert c.post(f"/api/jobs/{name}/cancel", headers=AUTH).json()["state"] == "done"
        r = c.post("/api/jobs", json={"clip": key, "dur": 5, "name": name}, headers=AUTH)
        assert r.status_code == 409  # work/<name> exists


@needs_ffmpeg
def test_stage_clip_copies_only_the_window_clip(tmp_path, teslacam):
    lib = footage.Library(tmp_path / "data", autodetect_fn=lambda: [teslacam])
    lib.scan(wait=True)
    clip = lib.all_clips()[0]
    out = footage.stage_clip(clip, tmp_path / "w" / "_src", tmp_path / "data", log=lambda *a: None)
    files = sorted(p.name for p in out.glob("*.mp4"))
    assert files == sorted(f"{clip['timestamp']}-{c}.mp4" for c in clip["cams"])
    idx = json.loads((out / "index.json").read_text())
    assert idx[0]["id"] == clip["id"] and all(p.startswith(str(out.resolve())) for p in idx[0]["cams"].values())

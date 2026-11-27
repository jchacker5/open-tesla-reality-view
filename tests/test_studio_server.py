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
    for site in ("cross-site", "same-site"):  # same-site: another port on 127.0.0.1
        r = client.get("/api/sources", headers={**AUTH, "Sec-Fetch-Site": site})
        assert r.status_code == 403 and site in r.json()["detail"]
    for site in ("same-origin", "none"):
        assert client.get("/api/sources", headers={**AUTH, "Sec-Fetch-Site": site}).status_code == 200


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


CITY_BASE = f"http://127.0.0.1:{PORT + 1}"


def make_city(cfg, name="drive1"):
    city = cfg.work / name / "viewer"
    city.mkdir(parents=True)
    (city / "scene.ply").write_bytes(b"ply\n" + b"\0" * 100)
    (city / "path.json").write_text("[]")
    return city


def test_cities_are_served_on_their_own_origin(tmp_path):
    """City pages come from a second server (another port, so another origin): no token, no cookie, their
    own CSP. The studio itself no longer serves /cities."""
    app, cfg, _ = make_app(tmp_path)
    make_city(cfg)
    studio = TestClient(app, base_url=BASE)
    studio.get(f"/?token={TOKEN}")
    assert studio.get("/cities/drive1/").status_code == 404

    c = TestClient(server.create_city_app(cfg), base_url=CITY_BASE)
    r = c.get("/cities/drive1/")  # no token needed
    assert r.status_code == 200 and "<html" in r.text.lower()
    csp = r.headers["content-security-policy"]
    assert csp != server.CSP and "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
    assert "cdn.jsdelivr.net" not in csp
    image_sources = next(d for d in csp.split(";") if d.strip().startswith("img-src"))
    assert server.MAP_TILES in image_sources  # opt-in map images work in the city viewer
    assert server.MAP_TILES not in server.CSP  # the studio itself still makes no external requests
    assert r.headers["x-frame-options"] == "DENY" and r.headers["referrer-policy"] == "no-referrer"
    assert c.get("/cities/drive1/path.json").json() == []
    assert c.get("/cities/drive1/vendor/three/three.module.min.js").status_code == 200
    assert c.get("/cities/drive1/vendor/spark/spark.module.min.js").status_code == 200
    assert "cdn.jsdelivr.net" not in server.viewer_html_text()
    r = c.get("/cities/drive1/scene.ply", headers={"Range": "bytes=0-3"})
    assert r.status_code == 206 and r.content == b"ply\n"
    assert c.head("/cities/drive1/scene.ply").status_code == 200  # viewer v2 probes it with HEAD
    assert c.head("/cities/drive1/").status_code == 200
    assert "<html" in c.get("/cities/drive1/index.html").text.lower()  # the viewer's own folder is a fallback
    assert c.get("/cities/drive1/../../secret").status_code == 404
    assert c.get("/cities/drive1/%2e%2e/%2e%2e/secret").status_code == 404
    assert c.get("/cities/nope/").status_code == 404
    r = c.get("/cities/drive1?t=3", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/cities/drive1/?t=3"
    for path in ("/", "/api/sources", "/static/app.js", f"/?token={TOKEN}"):
        assert c.get(path, headers=AUTH).status_code == 404  # nothing of the studio's lives here
    assert c.delete("/cities/drive1/scene.ply").status_code == 405
    assert c.get("/cities/drive1/", headers={"Host": f"evil.example:{PORT + 1}"}).status_code == 403
    assert c.get("/cities/drive1/", headers={"Host": f"127.0.0.1:{PORT}"}).status_code == 403


def test_city_pages_cannot_get_the_token(tmp_path):
    app, cfg, _ = make_app(tmp_path)
    make_city(cfg)
    c = TestClient(server.create_city_app(cfg), base_url=CITY_BASE)
    # the browser sends the studio's cookie to the cities port too (cookies ignore ports): never echoed or used
    for path in ("/cities/drive1/", "/cities/drive1/path.json", "/cities/drive1/scene.ply"):
        r = c.get(path, headers={"Cookie": f"{server.COOKIE}={TOKEN}"})
        assert TOKEN not in r.text and TOKEN not in str(r.headers) and "set-cookie" not in r.headers

    # and from a city page, the studio is same-site (same host, other port) and cross-origin: refused
    studio = TestClient(app, base_url=BASE)
    studio.get(f"/?token={TOKEN}")  # the browser holds the cookie
    from_city = {"Origin": CITY_BASE, "Sec-Fetch-Site": "same-site"}
    assert studio.get("/api/sources", headers={"Origin": CITY_BASE}).status_code == 403
    assert studio.get("/api/sources", headers={"Sec-Fetch-Site": "same-site"}).status_code == 403
    assert studio.post("/api/sources/rescan", headers={**AUTH, **from_city}).status_code == 403
    # a subresource load of the studio page (iframe, fetch) is refused; X-Frame-Options/CSP stop framing too
    r = studio.get("/", headers={"Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "no-cors"})
    assert r.status_code == 403 and TOKEN not in r.text
    r = studio.get("/", headers={"Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "navigate"})
    assert r.status_code == 200 and r.headers["x-frame-options"] == "DENY"  # a top-level visit is fine
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


def test_studio_page_points_at_the_cities_origin(client):
    client.get(f"/?token={TOKEN}")
    assert f'<meta name="otrv-cities" content="{CITY_BASE}">' in client.get("/").text


@pytest.mark.parametrize("has_drive", [False, True])
def test_viewer_features_gate_the_drive_button(client, monkeypatch, has_drive):
    page = "<html><script>window.otrvDrive = {};</script></html>" if has_drive else "<html>viewer v1</html>"
    monkeypatch.setattr(server, "viewer_html_text", lambda: page)
    assert client.get("/api/viewer", headers=AUTH).json() == {"base": CITY_BASE, "drive": has_drive}


def test_drive_button_uses_the_viewer_v2_deep_link():
    """Viewer v2 (#15) contract: drive mode is /cities/<name>/#cam=drive; the old ?mode=drive is gone, and
    the button is gated on /api/viewer's `drive`."""
    js = (server.STATIC / "app.js").read_text()
    assert 'cityUrl(name, "#cam=drive")' in js and "mode=drive" not in js
    assert "driveButton(job.name, viewerInfo.drive)" in js and "driveButton(c.name, drive)" in js
    assert 'api("/api/viewer")' in js


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


@pytest.mark.parametrize("dur", ["1e308", "99999", "0.0001"])
def test_estimate_clamps_dur(client, dur):
    r = client.get(f"/api/estimate?dur={dur}", headers=AUTH)
    assert r.status_code == 200, r.text
    est = r.json()
    assert 0 < est["images"] <= 60 * 10 * 9 and est["total"] > 0


@pytest.mark.parametrize(
    "query", ["dur=nan", "dur=inf", "dur=1e400", "dur=-1", "dur=0", "fps=0", "cams=0", "cams=99", "dur=x"]
)
def test_estimate_rejects_invalid(client, query):
    assert client.get(f"/api/estimate?{query}", headers=AUTH).status_code == 422


CLIP = "2026-01-02_03-04-05-front.mp4"


def uploads_left(lib):
    return sorted(p.name for p in lib.uploads.rglob("*") if p.is_file()) if lib.uploads.exists() else []


def test_upload_writes_the_clip(tmp_path):
    app, cfg, lib = make_app(tmp_path)
    c = TestClient(app, base_url=BASE)
    r = c.put(f"/api/uploads/RecentClips/{CLIP}", content=b"x" * 1000, headers=AUTH)
    assert r.status_code == 200 and r.json()["bytes"] == 1000
    assert uploads_left(lib) == [CLIP]


def test_upload_size_limit(tmp_path):
    cfg = server.Config(tmp_path / "data", tmp_path / "work", PORT, token=TOKEN, cwd=tmp_path, max_upload=100)
    lib = footage.Library(cfg.data, autodetect_fn=lambda: [])
    c = TestClient(server.create_app(cfg, library=lib), base_url=BASE)
    r = c.put(f"/api/uploads/RecentClips/{CLIP}", content=b"x" * 101, headers=AUTH)  # Content-Length says so
    assert r.status_code == 413 and "limit" in r.json()["detail"]

    def chunks():  # no Content-Length (chunked): the limit is enforced while streaming
        for _ in range(5):
            yield b"x" * 40

    r = c.put(f"/api/uploads/RecentClips/{CLIP}", content=chunks(), headers=AUTH)
    assert r.status_code == 413
    assert uploads_left(lib) == []  # no .part left behind
    assert c.put(f"/api/uploads/RecentClips/{CLIP}", content=b"x" * 100, headers=AUTH).status_code == 200


def test_upload_client_disconnect_removes_the_part_file(tmp_path):
    """The browser goes away mid-upload: no half-written .part (or clip) is left in uploads/."""
    import asyncio

    app, cfg, lib = make_app(tmp_path)
    messages = [
        {"type": "http.request", "body": b"x" * 100, "more_body": True},
        {"type": "http.disconnect"},
    ]
    sent = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "PUT",
        "scheme": "http",
        "path": f"/api/uploads/RecentClips/{CLIP}",
        "raw_path": f"/api/uploads/RecentClips/{CLIP}".encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", f"127.0.0.1:{PORT}".encode()), (b"x-otrv-token", TOKEN.encode())],
        "client": ("127.0.0.1", 5555),
        "server": ("127.0.0.1", PORT),
    }
    try:
        asyncio.run(app(scope, receive, send))
    except Exception:
        pass  # starlette.requests.ClientDisconnect, depending on the version
    assert uploads_left(lib) == []
    assert not any(m.get("status") == 200 for m in sent)


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

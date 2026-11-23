"""The studio HTTP server (FastAPI). Local only: binds 127.0.0.1, checks Host and Origin, and requires a
random per-launch token (in the URL it opens, Jupyter-style) on /api and /cities.

The token arrives once as /?token=...; the server swaps it for an HttpOnly SameSite=Strict cookie and
redirects to /. The page reads the token from a <meta> tag and sends it as X-OTRV-Token: state-changing
requests (POST/PUT/DELETE) must carry that header, so a cookie alone can't be used cross-site.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import shutil
import socket
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.datastructures import Headers, MutableHeaders

from .. import doctor, ingest, speed
from . import builds, footage, progress
from .jobs import TERMINAL, Runner, valid_name

STATIC = Path(__file__).parent / "static"
COOKIE = "otrv_studio"
CSP = (
    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self'; "
    "script-src 'self'; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)
# TODO(viewer, #13): the viewer (viewer/index.html, owned by feat/viewer-v2) still imports three.js and Spark
# from cdn.jsdelivr.net, so /cities/ pages can't carry the studio's `default-src 'self'` CSP yet. Once the
# viewer vendors those libraries, apply CSP to /cities/ too.
CITY_CSP = "object-src 'none'; base-uri 'none'; frame-ancestors 'self'"


class Config:
    def __init__(self, data: Path, work: Path, port: int, token: str | None = None, cwd: Path | None = None):
        self.data = Path(data).expanduser().resolve()
        self.work = Path(work)
        self.port = port
        self.token = token or secrets.token_urlsafe(24)
        self.cwd = cwd or Path.cwd()
        hosts = [f"127.0.0.1:{port}", f"localhost:{port}"]
        self.hosts = set(hosts)
        self.origins = {f"http://{h}" for h in hosts}


# ---- security middleware ----------------------------------------------------------------------------


class Guard:
    """Host/Origin checks, token auth for /api and /cities, and security headers on every response."""

    def __init__(self, app, cfg: Config):
        self.app, self.cfg = app, cfg

    def _token_ok(self, value: str | None) -> bool:
        return bool(value) and secrets.compare_digest(value.encode(), self.cfg.token.encode())

    def check(self, scope) -> tuple[int, str] | None:
        """None if the request may proceed, else (status, reason)."""
        h = Headers(scope=scope)
        if h.get("host") not in self.cfg.hosts:
            return 403, "bad Host header (the studio only answers on 127.0.0.1 / localhost)"
        origin = h.get("origin")
        if origin is not None and origin not in self.cfg.origins:
            return 403, "cross-origin request refused"
        if h.get("sec-fetch-site") == "cross-site":
            return 403, "cross-site request refused"
        path, method = scope["path"], scope["method"]
        if not (path.startswith("/api/") or path.startswith("/cities/") or path == "/cities"):
            return None
        header = h.get("x-otrv-token")
        if method not in ("GET", "HEAD"):
            return None if self._token_ok(header) else (401, "missing or wrong X-OTRV-Token")
        cookie = _cookies(h).get(COOKIE)
        query = parse_qs(scope.get("query_string", b"").decode()).get("token", [None])[0]
        if self._token_ok(header) or self._token_ok(cookie) or self._token_ok(query):
            return None
        return 401, "open the link `otrv studio` printed (it carries the access token)"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        bad = self.check(scope)
        if bad:
            resp = JSONResponse({"detail": bad[1]}, status_code=bad[0])
            return await resp(scope, receive, send)
        path = scope["path"]

        async def send_headers(message):
            if message["type"] == "http.response.start":
                hd = MutableHeaders(scope=message)
                hd["X-Content-Type-Options"] = "nosniff"
                hd["Referrer-Policy"] = "no-referrer"
                hd["Cross-Origin-Opener-Policy"] = "same-origin"
                if path.startswith("/cities"):
                    hd["Content-Security-Policy"] = CITY_CSP
                else:
                    hd["Content-Security-Policy"] = CSP
                    hd["X-Frame-Options"] = "DENY"
                if path.startswith("/api/") and "cache-control" not in hd:
                    hd["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_headers)


def _cookies(h: Headers) -> dict[str, str]:
    out = {}
    for part in h.get("cookie", "").split(";"):
        k, _, v = part.strip().partition("=")
        if k:
            out[k] = v
    return out


# ---- request models ---------------------------------------------------------------------------------


class SourceIn(BaseModel):
    path: str


class JobIn(BaseModel):
    clip: str
    start: float = Field(0, ge=0)
    dur: float = Field(20, gt=0, le=60)
    fps: float = Field(3, ge=0.5, le=10)
    steps: int = Field(15000, ge=100, le=100000)
    width: int = Field(1280, ge=320, le=1920)
    mask_movers: bool = False
    eval: bool = False
    global_mapper: bool = False
    name: str | None = None
    replace: bool = False


# ---- app --------------------------------------------------------------------------------------------


def create_app(cfg: Config, library: footage.Library | None = None, runner: Runner | None = None) -> FastAPI:
    lib = library or footage.Library(cfg.data)
    run = runner or Runner(
        cfg.work,
        plan=lambda job: builds.plan(job, cfg.data),
        units=builds.units,
        probe=builds.probe,
        cwd=cfg.cwd,
    )

    @asynccontextmanager
    async def lifespan(_app):
        run.start()
        lib.scan()
        yield
        await asyncio.to_thread(run.stop, 8)

    app = FastAPI(title="otrv studio", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.cfg, app.state.library, app.state.runner = cfg, lib, run
    doctor_cache: dict = {"t": 0.0, "checks": None}

    # ---- pages ----

    @app.get("/", include_in_schema=False)
    def index(request: Request, token: str | None = None):
        if token is not None:
            if not secrets.compare_digest(token.encode(), cfg.token.encode()):
                return HTMLResponse(_locked_page(), status_code=401)
            resp = RedirectResponse("/", status_code=303)
            resp.set_cookie(COOKIE, cfg.token, httponly=True, samesite="strict", path="/")
            return resp
        cookie = request.cookies.get(COOKIE)
        if not cookie or not secrets.compare_digest(cookie.encode(), cfg.token.encode()):
            return HTMLResponse(_locked_page(), status_code=401)
        html = (STATIC / "index.html").read_text().replace("{{TOKEN}}", cfg.token)
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # ---- doctor ----

    @app.get("/api/doctor")
    def get_doctor(refresh: bool = False):
        if refresh or doctor_cache["checks"] is None or time.time() - doctor_cache["t"] > 120:
            checks = doctor.run_checks()
            doctor_cache.update(t=time.time(), checks=[c.__dict__ for c in checks])
        return {"checks": doctor_cache["checks"], "platform": sys.platform}

    # ---- sources ----

    def sources_payload():
        return {"sources": lib.sources(), "scan": lib.scan_state, "data": str(cfg.data)}

    @app.get("/api/sources")
    def get_sources():
        return sources_payload()

    @app.post("/api/sources")
    def add_source(body: SourceIn):
        try:
            lib.add_source(body.path)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return sources_payload()

    @app.delete("/api/sources")
    def remove_source(path: str):
        lib.remove_source(path)
        return sources_payload()

    @app.post("/api/sources/pick")
    def pick_source():
        if sys.platform != "darwin":
            raise HTTPException(501, "the folder picker needs macOS; paste the folder path instead")
        path = footage.pick_folder()
        if not path:
            return {"path": None, **sources_payload()}
        try:
            lib.add_source(path)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"path": path, **sources_payload()}

    @app.post("/api/sources/rescan")
    def rescan():
        lib.scan()
        return sources_payload()

    @app.put("/api/uploads/{rel:path}")
    async def upload(rel: str, request: Request):
        """Fallback for browsers without a folder picker path: one clip file per request, raw body."""
        parts = [p for p in rel.split("/") if p]
        if not parts or not ingest.CLIP_RE.match(parts[-1]) or any(p in (".", "..") for p in parts):
            raise HTTPException(400, "only TeslaCam clip files (YYYY-MM-DD_HH-MM-SS-<camera>.mp4) can be uploaded")
        kind = next((k for k in ingest.KINDS if k in parts), "Uploads")
        folder = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (parts[-2] if len(parts) > 1 else kind))
        dest_dir = lib.uploads / kind if folder in (kind, "TeslaCam") else lib.uploads / kind / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / parts[-1]
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        tmp.replace(dest)
        return {"path": str(dest), "bytes": dest.stat().st_size}

    # ---- drives & clips ----

    @app.get("/api/drives")
    def get_drives():
        return {"drives": [footage.card(d) for d in lib.drives()], "scan": lib.scan_state}

    def need_drive(drive_id: str) -> dict:
        d = lib.drive(drive_id)
        if not d:
            raise HTTPException(404, f"no drive {drive_id}")
        return d

    @app.get("/api/drives/{drive_id}")
    def get_drive(drive_id: str):
        return need_drive(drive_id)

    @app.get("/api/drives/{drive_id}/suggest")
    def get_suggest(drive_id: str, dur: int = Query(20, ge=5, le=60), lo: float = 10, hi: float = 40):
        d = need_drive(drive_id)
        clips = [lib.clip(c["key"]) for c in d["clips"]]
        return {"suggestions": footage.suggest(d, [c for c in clips if c], dur, lo, hi)}

    def need_clip(key: str) -> dict:
        c = lib.clip(key)
        if not c:
            raise HTTPException(404, f"no clip {key}")
        return c

    @app.get("/api/clips/{key}")
    def get_clip(key: str):
        return need_clip(key)

    @app.get("/api/clips/{key}/thumb.jpg")
    def get_thumb(key: str, cam: str = "front", t: float | None = None):
        need_clip(key)
        p = lib.thumb(key, cam, t)
        if not p:
            raise HTTPException(404, "no thumbnail (camera missing or ffmpeg failed)")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/clips/{key}/video/{cam}")
    def get_video(key: str, cam: str):
        c = need_clip(key)
        if cam not in c["cams"]:
            raise HTTPException(404, f"clip has no {cam} camera")
        return FileResponse(c["cams"][cam], media_type="video/mp4", headers={"Cache-Control": "private, max-age=3600"})

    # ---- estimates & jobs ----

    @app.get("/api/estimate")
    def get_estimate(cams: int = 4, dur: float = 20, fps: float = 3, steps: int = 15000):
        rates = progress.history_rates(cfg.work)
        n = max(1, round(dur * fps)) * cams
        per = progress.estimate_build(rates, n, steps)
        return {"stages": per, "total": round(sum(per.values()), 1), "images": n, "history": sorted(rates)}

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [j.to_dict() for j in run.list()], "current": run.current.name if run.current else None}

    def need_job(name: str):
        j = run.get(name)
        if not j:
            raise HTTPException(404, f"no build {name}")
        return j

    @app.post("/api/jobs")
    def create_job(body: JobIn):
        clip = need_clip(body.clip)
        if len(clip["cams"]) < 4:
            raise HTTPException(400, f"clip {clip['id']} has {len(clip['cams'])} cameras; a build needs 4+")
        secs = clip.get("seconds")
        if secs and body.start + body.dur > secs + 0.5:
            raise HTTPException(400, f"window {body.start:g}+{body.dur:g}s runs past the end of the clip ({secs:g}s)")
        if body.name:
            name = body.name
            if not valid_name(name):
                raise HTTPException(400, "name: letters, digits, '.', '_' and '-' only (max 80)")
            if run.busy(name):
                raise HTTPException(409, f"{name} is already building")
            if (cfg.work / name).exists() and not body.replace:
                raise HTTPException(409, f"work/{name} exists; pick another name or set replace")
        else:
            base = speed.run_name(clip["id"])
            if body.start:
                base += f"_{int(body.start)}s"
            name, i = base, 2
            while (cfg.work / name).exists() or run.get(name):
                name, i = f"{base}-{i}", i + 1
        spec = body.model_dump(exclude={"name", "replace", "clip"})
        spec["clip"] = {k: clip[k] for k in ("key", "id", "timestamp", "kind", "seconds", "cams", "event", "mph")}
        spec["clip"]["mph_1hz"] = clip.get("mph_1hz")
        try:
            job = run.submit(name, spec)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        return job.to_dict()

    @app.get("/api/jobs/{name}")
    def get_job(name: str):
        return need_job(name).to_dict()

    @app.post("/api/jobs/{name}/cancel")
    def cancel_job(name: str):
        need_job(name)
        return run.cancel(name).to_dict()

    @app.post("/api/jobs/{name}/resume")
    def resume_job(name: str):
        need_job(name)
        try:
            return run.resume(name).to_dict()
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/api/jobs/{name}/events", response_class=EventSourceResponse)
    async def job_events(name: str, request: Request, after: int = 0):
        need_job(name)
        try:
            after = max(after, int(request.headers.get("last-event-id") or 0))
        except ValueError:
            pass
        path = run.events_path(name)
        offset, buf, last_ping = 0, b"", time.monotonic()
        yield ServerSentEvent(retry=2000, comment="otrv")
        while True:
            chunk = b""
            if path.exists():
                with open(path, "rb") as f:
                    f.seek(offset)
                    chunk = f.read()
                offset += len(chunk)
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("id", 0) > after:
                    after = ev["id"]
                    yield ServerSentEvent(raw_data=line.decode(), id=str(ev["id"]))
            job = run.get(name)
            if not chunk and (job is None or job.state in TERMINAL):
                yield ServerSentEvent(event="end", raw_data="{}")
                return
            if await request.is_disconnected():
                return
            if time.monotonic() - last_ping > 15:
                last_ping = time.monotonic()
                yield ServerSentEvent(comment="ping")
            await asyncio.sleep(0.25)

    # ---- cities ----

    def need_city(name: str) -> Path:
        d = cfg.work / name
        if not valid_name(name) or not builds.is_city(d):
            raise HTTPException(404, f"no city {name}")
        return d

    @app.get("/api/cities")
    def list_cities():
        return {"cities": builds.cities(cfg.work)}

    @app.get("/api/cities/{name}/poster.jpg")
    def city_poster(name: str):
        p = builds.poster(need_city(name))
        if not p:
            raise HTTPException(404, "no poster")
        return FileResponse(p, media_type="image/jpeg")

    @app.delete("/api/cities/{name}")
    def delete_city(name: str):
        d = need_city(name)
        if run.busy(name):
            raise HTTPException(409, f"{name} is building; cancel it first")
        run.forget(name)
        shutil.rmtree(d)
        return {"deleted": name}

    @app.get("/cities/{name}", include_in_schema=False)
    def city_redirect(name: str, request: Request):
        q = request.url.query
        return RedirectResponse(f"/cities/{name}/" + (f"?{q}" if q else ""), status_code=307)

    @app.get("/cities/{name}/", include_in_schema=False)
    def city_page(name: str):
        need_city(name)
        from ..__main__ import viewer_html

        return HTMLResponse(viewer_html().read_text(), headers={"Cache-Control": "no-store"})

    @app.get("/cities/{name}/{rel:path}", include_in_schema=False)
    def city_file(name: str, rel: str):
        root = (need_city(name) / "viewer").resolve()
        p = (root / rel).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(p)

    return Guard(app, cfg)


def _locked_page() -> str:
    return (STATIC / "locked.html").read_text()


def free_port(start: int, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit(f"no free port in {start}-{start + tries - 1}; pass --port")


def serve(data: str, port: int = 8770, open_browser: bool = True, work: str = "work") -> None:
    import uvicorn

    port = free_port(port)
    cfg = Config(Path(data), Path(work).resolve(), port)
    app = create_app(cfg)
    url = f"http://127.0.0.1:{port}/?token={cfg.token}"
    print(f"otrv studio: {url}")
    print("  local only (127.0.0.1); footage is read in place and never uploaded. ctrl-c to stop.", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)

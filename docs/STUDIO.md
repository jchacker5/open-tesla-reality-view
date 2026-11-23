# otrv studio

`otrv studio` is a local web app over the same pipeline as the CLI. You pick your TeslaCam footage, see
your drives, choose a 20-60 s window and build a Gaussian-splat "city" with live progress. Then you open
it in the replay viewer, or in **Drive** mode.

```sh
uv sync --extra studio          # FastAPI + uvicorn (small; no PyTorch)
uv run otrv studio              # opens http://127.0.0.1:8770/?token=...
uv run otrv --data /path studio # footage dir other than data/teslacam (same as $OTRV_DATA)
```

It needs the same tools as the CLI (`otrv doctor`): ffmpeg, COLMAP 4.0+ and Brush. The Sources page
shows their status.

## Pages

| Page | What it does |
|---|---|
| **Sources** | The privacy banner, `doctor` status chips, and where your footage is: a USB drive (any `/Volumes/*/TeslaCam`, found automatically), folders you add by path or with **Choose folder…** (the macOS folder picker), the `otrv ingest` data folder, and uploads. |
| **Drives** | Consecutive clips grouped into drives. Each card shows four camera thumbnails, the date, length, mean speed, a speed sparkline and the GPS route from the clips' SEI telemetry (drawn as an SVG line, no map tiles), plus a **Best window** badge from `otrv suggest`'s scoring. |
| **Drive detail** | The front camera with the other cameras synced below it. A speed strip over the whole drive has a draggable build window (20-60 s, kept inside one clip). It also shows suggested windows, a build-time estimate and **Build**. **Advanced** sets fps, training steps, YOLO mover masking, held-out scoring (PSNR) and the name. |
| **Build** | A FIFO queue (one GPU, one build at a time). Each build has a stage stepper (frames → masks → features → match → map → undistort → train → export) with a progress bar per stage, the overall ETA, a log tail, and **Cancel** / **Resume**. |
| **Gallery** | Finished cities: a poster (first front frame), date, splat size, disk use and PSNR, with **Drive** (`/cities/<name>/?mode=drive`), **Replay** and **Delete**. |

## Privacy and security

- **Local only.** The server binds `127.0.0.1` and answers only to `Host: 127.0.0.1:<port>` or
  `localhost:<port>`, which blocks DNS rebinding. It refuses requests with a foreign `Origin` or with
  `Sec-Fetch-Site: cross-site`.
- **A token per launch**, Jupyter-style. The printed URL carries `?token=…`. The server swaps it for an
  `HttpOnly; SameSite=Strict` cookie and redirects to `/`. `/api/*` and `/cities/*` need the token as a
  header, the cookie, or `?token=`. Any request that changes state (POST/PUT/DELETE) must send the
  `X-OTRV-Token` header, so a cookie alone can't be used for CSRF.
- **No external requests.** The studio's pages are plain HTML/CSS/JS with no third-party code and no
  CDN. They are served with `Content-Security-Policy: default-src 'self'` and `frame-ancestors 'none'`,
  and a test checks that the static files reference no `http(s)://` URLs.
- **Footage is indexed in place.** Nothing is copied until you press Build. Then only the chosen clip's
  camera files (about 200 MB, not the drive's 100 GB) are copied to `work/<name>/_src/`, and that copy
  is deleted when the build succeeds.
- **Known gap.** The replay viewer (`viewer/index.html`) still loads three.js and Spark from
  cdn.jsdelivr.net, so `/cities/<name>/` pages don't get the strict CSP yet. Once the viewer vendors
  those libraries (issue #13), `CITY_CSP` in `tsplat/studio/server.py` should become `default-src 'self'` too.

## How a build runs

`POST /api/jobs` queues a build. The single worker thread runs it as phases:

1. **prep**: copy the clip's camera files to `work/<name>/_src/` (with a one-clip `index.json` and
   any `egomask/`), then `otrv --data work/<name>/_src prep <clip> --name <name> --start … --dur … --fps …`
2. **recon**: `otrv recon <name> --steps …` (with `--skip-sfm` when resuming after undistort)
3. **bench**: `otrv bench <name> --note studio` (optional: a failure is logged, not fatal)
4. **finalize**: poster, delete `_src/`, write `city.json`

Each command runs in its own session (`start_new_session=True`) under a PTY (`pty.openpty`), so Brush
draws its progress bar. Output is split on `\r` and `\n`, ANSI escapes are stripped, and each line is
matched against:

| Stage | Regex |
|---|---|
| features | `Processed file \[(\d+)/(\d+)\]` |
| match | `Processing block \[(\d+)/(\d+)\]` |
| map | `Registering image #\d+ \(num_reg_frames=(\d+)\)`, against the known frame count (monotonic) |
| undistort | `Undistorting image \[(\d+)/(\d+)\]` (two passes, images then masks, shown as one bar) |
| train | `(\d+)/(\d+)\s+Steps \(([\d.]+)/s, (.+?) remaining\)` |

Stage starts come from the lines `otrv` prints (`[name] N cams`, `ego mask:`, `$ colmap <cmd>`,
`$ …brush…`, `splat -> `). Frames and masks print no progress, so the runner counts the files written
so far.

Files in `work/<name>/`:

| File | Contents |
|---|---|
| `job.json` | spec and state. A build that was running when the studio stopped comes back as *interrupted*. |
| `events.jsonl` | one event per line, with ids that keep increasing across resumes. `GET /api/jobs/<name>/events` streams it as SSE and honours `Last-Event-ID`, so a reconnecting browser replays only what it missed. |
| `stages/<stage>.done` | `{"seconds", "units"}`. Resume skips every phase whose stages are all done. |
| `studio.log` | raw output |

**Cancel** sends SIGTERM to the process group (otrv and its colmap/brush children), then SIGKILL after
5 s. **Resume** re-queues a failed, cancelled or interrupted build.

**ETA** is the current stage's own estimate (Brush's, or the progress rate so far) plus a prior for
each remaining stage. The priors are this Mac's median seconds per image (or per training step) from
past `stages/*.done` markers and `timings.json`, with built-in defaults until there is history.

## API

All under `/api` need the token (see above).

```
GET  /                                   the app (token -> cookie)
GET  /api/doctor
GET  /api/sources        POST /api/sources {path}        DELETE /api/sources?path=
POST /api/sources/pick   POST /api/sources/rescan        PUT /api/uploads/<folder>/<clip file>
GET  /api/drives         GET /api/drives/{id}            GET /api/drives/{id}/suggest?dur=20
GET  /api/clips/{key}    GET /api/clips/{key}/thumb.jpg?cam=front    GET /api/clips/{key}/video/{cam} (byte ranges)
GET  /api/estimate?cams=4&dur=20&fps=3&steps=15000
POST /api/jobs {clip, start, dur, fps, steps, mask_movers, eval, name}
GET  /api/jobs           GET /api/jobs/{id}              GET /api/jobs/{id}/events (SSE)
POST /api/jobs/{id}/cancel                               POST /api/jobs/{id}/resume
GET  /api/cities         DELETE /api/cities/{name}       GET /api/cities/{name}/poster.jpg
GET  /cities/{name}/     the viewer, serving work/<name>/viewer/ (byte ranges)
```

The studio keeps its own state in `<data>/studio/`: `sources.json`, `cache.json` (per-clip length,
speed and route, keyed by path + size + mtime, so a rescan reads only new clips) and `thumbs/`.
Uploads go to `<data>/uploads/`.

## Trying it without a car

```sh
uv run python -m tsplat.studio.synth /tmp/TeslaCam   # 3 synthetic drives (testsrc2 video + SEI telemetry)
uv run otrv studio                                   # then add /tmp/TeslaCam on the Sources page
```

The synthetic clips carry real Tesla-format SEI messages (speed, GPS, heading), so the drives, routes
and best windows behave like real footage. A test pattern won't reconstruct into a city, but prep,
features and matching run and show real progress.

## Code

| Path | Purpose |
|---|---|
| `tsplat/studio/progress.py` | regexes, stage tracker, ETA priors (standard library only) |
| `tsplat/studio/jobs.py` | FIFO runner, PTY subprocesses, events.jsonl, cancel/resume (standard library only) |
| `tsplat/studio/footage.py` | sources, in-place indexing, drives, thumbnails, the per-job clip copy |
| `tsplat/studio/builds.py` | the phases a build runs; cities |
| `tsplat/studio/server.py` | FastAPI app, security middleware, SSE |
| `tsplat/studio/static/` | the front end (no build step) |
| `tsplat/studio/synth.py` | synthetic TeslaCam generator |

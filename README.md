<div align="center">

# Open Tesla Reality View

**Your Tesla's dashcam footage, rebuilt as a 3D world you can fly through.**

Pull the USB drive from your glovebox, run `ingest`, `prep`, `recon` and `view`, and replay your drive as a
[3D Gaussian splat](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/): a chase cam behind
your car, every camera feed beside it, and speed, steering and Autopilot state from the telemetry
Tesla saves in each clip. Runs locally on an Apple-silicon Mac. No cloud, no account, your footage
never leaves your machine.

[![CI](https://github.com/jchacker5/open-tesla-reality-view/actions/workflows/ci.yml/badge.svg)](https://github.com/jchacker5/open-tesla-reality-view/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-3d7bff)](LICENSE)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-3ecf8e)](CONTRIBUTING.md)

<!-- media: demo video + screenshots land here (docs/media/) -->

</div>

> **Status: early and moving fast.** It works end to end on real footage from a 4-camera (HW3)
> car. We're building in public, so issues, ideas and PRs are welcome. See [Roadmap](#roadmap).

## Features

- **Works with your car's drive as-is.** Reads the `TeslaCam` folder (RecentClips, SavedClips,
  SentryClips). 4-camera (front, back, repeaters) and 6-camera (HW4 with pillar cams) cars are both
  supported.
- **Rig-aware reconstruction.** All cameras are solved together as one rigid rig in COLMAP, starting
  from an approximate Tesla camera layout that bundle adjustment refines. This keeps every camera
  in a single model.
- **Clean scenes.** Your own car's body is masked automatically using many clips. Moving cars and
  people can be masked with YOLO segmentation so they don't leave smears.
- **Gaussian splat training on your Mac's GPU** with [Brush](https://github.com/ArthurBrussee/brush)
  (Metal).
- **Replay viewer** ([Spark](https://sparkjs.dev) + three.js): chase, driver, drone and free-fly
  cameras, the raw camera feeds synced on the side, a telemetry HUD, a GPS street map, and a
  drive mode where you take the wheel inside the reconstruction (keyboard, gamepad or touch).
- **Telemetry** decoded from the SEI metadata in every clip (firmware 2025.44.25+), following
  [teslamotors/dashcam](https://github.com/teslamotors/dashcam): speed, gear, steering, brake, turn
  signals, Autopilot state, GPS. The decoder is pure Python, with no protobuf dependency.
- **Ready for generative view synthesis.** `otrv nvs-export` packages posed frames and the
  chase-cam path for diffusion novel-view models such as
  [ViewNeXt](https://phai-lab.github.io/ViewNext/).

## Quick start

```sh
# 1. tools + this repo (macOS, Apple silicon)
brew install uv ffmpeg colmap            # COLMAP 4.0+ (rig support)
git clone https://github.com/jchacker5/open-tesla-reality-view && cd open-tesla-reality-view
uv sync
uv run otrv setup-brush                  # downloads Brush v0.3.0, verifies its sha256, installs to ~/.local/opt/brush
uv run otrv doctor                       # checks every tool; prints the exact fix for anything missing

# 2. your footage: plug the car's USB drive into the Mac
uv run otrv ingest                       # copies RecentClips + SavedClips -> data/teslacam (resumable)
uv run otrv list                         # clips with 4+ cameras and their mean speed
uv run otrv suggest                      # best 20 s windows of 10-40 mph, as ready-to-paste prep commands

# 3. rebuild 20 seconds of driving
uv run otrv prep 2026-11-20_07-20-00 --name drive1 --start 10 --dur 20
uv run otrv recon drive1                 # SfM + splat training: the slow step (see Costs below)
uv run otrv view drive1                  # opens the replay in your browser
```

Every command has `--help` with defaults, units and time costs. `ingest`, `prep` and `recon` run a
quick `doctor` check first and stop with the fix if a tool is missing; errors print one line
(add `--debug` for the traceback).

**Viewing needs internet.** The viewer loads three.js and Spark from a CDN. Your footage and splat
stay on your machine; only those libraries are fetched.

Already have footage on another disk? `uv run otrv ingest --src /path/to/TeslaCam`, or keep the
footage somewhere else entirely with `--data /path` (or `export OTRV_DATA=/path`): ingest, list,
suggest, egomask and prep all use it. The tool only reads from the source; nothing on the car's
drive is modified. `ingest --since 2026-11-01` copies only recent clips, and Sentry clips (recorded
while parked) are opt-in: `--kinds RecentClips,SavedClips,SentryClips`. Before copying, ingest
prints the size and your free space, and stops if it won't fit.

**Optional extras.**
- `uv run otrv egomask`: builds better masks of your own car's body from 8+ driving clips (run
  once). Without it, `prep` guesses the mask from the window, which works but is rougher.
- `prep --mask-movers`: masks moving cars and people with YOLO so they don't leave smears. Needs
  `uv sync --extra movers` first, which downloads PyTorch (~1 GB).

**Getting good results.** Pick 15-30 s at 10-40 mph past buildings, trees and signs, in daylight
(`otrv suggest` finds these from telemetry). Highways, night and parked clips reconstruct poorly.
`--fps 4` helps for slow driving. `recon --steps 30000` gives the best quality, and `recon --live`
opens Brush's live training window.

### Costs

| Step | Cost |
|---|---|
| `ingest` | Disk: whatever you copy. A full hour of RecentClips from 4-6 cameras is several GB; ingest prints the exact size first. Narrow it with `--since` / `--kinds`. |
| `uv sync --extra movers` | ~1 GB download (PyTorch + ultralytics). Only needed for `--mask-movers`. |
| `prep` (20 s at 3 fps) | ~1-2 min. 240 images on a 4-camera car, 360 on a 6-camera car. |
| `recon` (20 s at 3 fps, 4 cameras = 240 images) | Training dominates: Brush at 30000 steps took **73 min on an M4 with 16 GB**. Time scales roughly with `--steps`, so the default 15000 should take about half that; COLMAP adds a few minutes. |
| `work/<name>/` | A few GB per run (frames, masks, dense model, splat checkpoints). |

## How it works

```
TeslaCam/ ──ingest──▶ index.json ──prep──▶ frames · masks · rig prior · pairs · telemetry
                                                │
                                   recon        ▼
                    COLMAP SIFT ─▶ rig configurator ─▶ mapper ─▶ undistort ─▶ Brush (Metal)
                                                                     │            │
                                                     chase-cam path ◀┘            ▼ scene.ply
                                                                        view: Spark + three.js
```

| Step | What happens |
|---|---|
| `doctor` | check OS, ffmpeg, COLMAP 4.0+ (with `rig_configurator`), Brush 0.3.x, disk and RAM; print fixes |
| `setup-brush` | download the Brush v0.3.0 release for your platform, verify its published sha256, install it |
| `ingest` | rsync the chosen `TeslaCam` folders, group per-camera files into clips, read each clip's mean speed from its telemetry |
| `list` / `suggest` | show clips with speed; rank 20 s windows by sustained 10-40 mph and print `prep` commands |
| `egomask` | optional: find pixels that look identical across many different drives: that's your car's body |
| `prep` | sample frames, mask the car (and optionally moving objects), write a rig-aware match list (same camera over time + overlapping neighbours) and the approximate rig, decode telemetry, cut the side-panel clips |
| `recon` | COLMAP features → rig configurator → mapper → undistort → Brush training → trajectory, with real-world scale from telemetry speed |
| `view` | local server with byte-range support, so the synced videos can seek (three.js/Spark come from a CDN) |
| `nvs-export` | posed frames + chase-cam target poses for a novel-view diffusion model |
| `bench` | score a run (per-camera registration, reprojection, held-out PSNR, timings), gate it, log it to `work/ledger.jsonl`; see [docs/HILL_CLIMB.md](docs/HILL_CLIMB.md) |

## The viewer

`viewer/` is a static page (`index.html` + `car.js`, three.js and Spark from a CDN). Serve the
folder that holds `scene.ply` and `path.json` (`otrv view` does this) and open it in a browser.

| Key | Action | Key | Action |
|---|---|---|---|
| `Space` | play / pause (handbrake while driving) | `←` `→` | ∓2 s (steer while driving) |
| `1` `2` `3` | chase, driver, free camera | `5` | drone camera over the street map |
| `4` | **drive** the car yourself | `R` / double-click | reset the view |
| `U` | hide the UI | `F` / `M` | fullscreen / mute |

- **Cameras.** Drag to orbit and wheel or pinch to zoom; follow cameras spring back after 2 s.
  In free mode, right-drag, Shift-drag or two fingers pan.
- **Drive mode** (`4`, the DRIVE button or gamepad Start). An arcade car model at 120 Hz. Keyboard:
  W/A/S/D or arrows, Shift boost, Space handbrake, H horn, C look back, V camera. Gamepads with the
  standard mapping: RT/LT throttle and brake/reverse, left stick steers, A handbrake, B horn,
  Y camera, right stick looks around. On phones there's a joystick plus gas, brake and handbrake
  buttons. The car stays near the recorded road, where the splat exists. An optional
  `collision.json` next to `scene.ply` adds real ground height and walls.
- **Sound** is synthesised in the browser: motor whine, wind, tyres, horn and a reverse chime.
  It starts on your first click or key press. You can use your own recordings instead, see
  [viewer/sounds/README.md](viewer/sounds/README.md).
- **Your car.** By default the viewer draws a built-in white crossover. With a glTF model in
  `viewer/models/` (`otrv car fetch` installs one, `otrv view` copies it) the real car is shown,
  repainted, with working lamps, blinkers and wheels, and its credit is shown at the bottom-left.
  See [viewer/models/README.md](viewer/models/README.md). Models are never committed.
- **Map (opt-in).** With GPS in the clip, the minimap always shows the route on a dark background.
  The `map` button (or `?map=1`) adds Esri's dark street tiles to the minimap and under the drone
  camera. It is off by default because tile requests reveal the drive's approximate location to
  Esri, and the viewer says so the first time you turn it on. `?map=0` forces it off.
- **Camera column.** Click a camera tile to see the reconstruction from that camera's position on
  the car. `C` or `hide` collapses the column.
- **Sharing.** `link` copies a `#t=…&cam=…` deep link, `rec` records the 3D view to a `.webm`
  file. Use `?ui=0`, `?clean=1` and `?aspect=9:16` for clean recordings. `?render=1` exposes
  `window.otrvRender` for frame-exact offline rendering (see the comment block in `index.html`).

## Requirements

- macOS on Apple silicon (M1 or newer). Brush publishes no Intel-Mac build. Linux and Windows
  x86_64 have Brush builds but are untested; see the [roadmap](#roadmap).
- 16 GB+ RAM recommended; a 20 s clip uses a few GB of disk in `work/`.
- Python 3.11+ via [uv](https://docs.astral.sh/uv/), ffmpeg, **COLMAP 4.0+** (rig support), Brush
  v0.3.0 (`otrv setup-brush`, or set `BRUSH=/path/to/brush_app`). `otrv doctor` checks all of it.
- Internet access to open the viewer (three.js and Spark load from a CDN).
- Telemetry needs Tesla firmware 2025.44.25+ on HW3/HW4. Older clips still reconstruct, just
  without the HUD.

## Roadmap

- [ ] Dynamic actors: moving cars and buses as their own splats (drivestudio / OmniRe, CUDA)
- [ ] Linux + CUDA training backend ([gsplat](https://github.com/nerfstudio-project/gsplat))
- [ ] Generative clean-up pass with ViewNeXt once its weights are public
- [ ] Multi-clip stitching for longer drives
- [ ] Export: share-able web build, video render of a camera path
- [ ] Windows support

Want one of these? Comment on its issue or open a draft PR. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Privacy

Dashcam footage contains GPS traces, licence plates, faces and where you live. This tool runs
locally and never uploads anything. `data/`, `work/`, footage and splats are git-ignored. If you
share a replay, consider what's in it.

## Credits

Built on [COLMAP](https://colmap.github.io/), [Brush](https://github.com/ArthurBrussee/brush),
[Spark](https://sparkjs.dev), [three.js](https://threejs.org) and Tesla's published
[dashcam metadata format](https://github.com/teslamotors/dashcam). Inspired by
[@ryanvogel's reconstructed-reality demo](https://x.com/ryanvogel/status/2124464850550292638).

Community project, **not affiliated with or endorsed by Tesla, Inc.** "Tesla" is a trademark of
Tesla, Inc. MIT licensed.

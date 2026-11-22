<div align="center">

# Open Tesla Reality View

**Your Tesla's dashcam footage, rebuilt as a 3D world you can fly through.**

Pull the USB drive from your glovebox, run five commands, and replay your drive as a
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
- **Replay viewer** ([Spark](https://sparkjs.dev) + three.js): chase, driver and free-fly cameras,
  the raw camera feeds synced on the side, and a telemetry HUD.
- **Telemetry** decoded from the SEI metadata in every clip (firmware 2025.44.25+), following
  [teslamotors/dashcam](https://github.com/teslamotors/dashcam): speed, gear, steering, brake, turn
  signals, Autopilot state, GPS. The decoder is pure Python, with no protobuf dependency.
- **Ready for generative view synthesis.** `otrv nvs-export` packages posed frames and the
  chase-cam path for diffusion novel-view models such as
  [ViewNeXt](https://phai-lab.github.io/ViewNext/).

## Quick start

```sh
# 1. tools (macOS, Apple Silicon)
brew install uv ffmpeg colmap
mkdir -p ~/.local/opt/brush && cd ~/.local/opt/brush
curl -LO https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-aarch64-apple-darwin.tar.xz
tar xf brush-app-aarch64-apple-darwin.tar.xz && cd -

# 2. this repo
git clone https://github.com/jchacker5/open-tesla-reality-view && cd open-tesla-reality-view
uv sync

# 3. your footage: plug the car's USB drive into the Mac
uv run otrv ingest                       # copies /Volumes/*/TeslaCam -> data/teslacam (resumable)
uv run otrv list                         # clips with 4+ cameras
uv run otrv egomask                      # once: masks your car's body (needs 8+ clips)

# 4. pick 20 seconds of driving and rebuild it
uv run otrv prep 2026-11-20_07-20-00 --name drive1 --start 10 --dur 20 --mask-movers
uv run otrv recon drive1                 # SfM + splat training: ~20-40 min on an M-series Mac
uv run otrv view drive1                  # opens the replay in your browser
```

Already have footage on another disk? `uv run otrv ingest --src /path/to/TeslaCam`, or symlink it
to `data/teslacam` and run `uv run otrv ingest --index-only`. The tool only reads from the source.
Nothing on the car's drive is modified.

**Getting good results.** Pick 15-30 s at 10-40 mph past buildings, trees and signs, in daylight.
Highways, night and parked clips reconstruct poorly. `--fps 4` helps for slow driving.
`recon --steps 30000` gives the best quality, and `recon --live` opens Brush's live training window.

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
| `ingest` | rsync the drive's `TeslaCam` folder, group per-camera files into clips |
| `egomask` | find pixels that look identical across many different drives: that's your car's body |
| `prep` | sample frames, mask the car (and optionally moving objects), write a rig-aware match list (same camera over time + overlapping neighbours) and the approximate rig, decode telemetry, cut the side-panel clips |
| `recon` | COLMAP features → rig configurator → mapper → undistort → Brush training → trajectory, with real-world scale from telemetry speed |
| `view` | local server with byte-range support, so the synced videos can seek |
| `nvs-export` | posed frames + chase-cam target poses for a novel-view diffusion model |
| `bench` | score a run (per-camera registration, reprojection, held-out PSNR, timings), gate it, log it to `work/ledger.jsonl`; see [docs/HILL_CLIMB.md](docs/HILL_CLIMB.md) |

## Requirements

- macOS on Apple Silicon (M1 or newer). Linux and Windows are on the [roadmap](#roadmap).
- 16 GB+ RAM recommended; a 20 s clip uses a few GB of disk in `work/`.
- Python 3.11+ via [uv](https://docs.astral.sh/uv/), ffmpeg, COLMAP 3.12+ (rig support), Brush v0.3.0.
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

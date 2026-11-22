# tesla-splat

Turn your Tesla's 6-camera dashcam into a 3D Gaussian-splat replay you can fly through: a chase cam
behind your car, the six raw camera feeds beside it, and speed, gear, Autopilot state, steering and
turn signals from the telemetry Tesla embeds in each clip.

Runs on an Apple-silicon Mac: COLMAP (CPU) for camera poses, [Brush](https://github.com/ArthurBrussee/brush)
(Metal) for splat training, [Spark](https://sparkjs.dev) + three.js for the viewer.

## 1. Get the footage off the car

**Best: the USB drive (glovebox).** It holds every clip, all 6 cameras, at full quality.
Plug it into the Mac and run:

```sh
uv run tsplat ingest          # finds /Volumes/*/TeslaCam, rsyncs to data/teslacam, builds index
uv run tsplat list            # clips that have all 6 cameras
```

Re-running `ingest` only copies new files. `RecentClips` is a rolling buffer of about the last hour, so
save a drive first: tap the dashcam icon or honk if you set that up, or ingest right after you park.

**Without touching the drive: Tesla app → Dashcam Viewer → download.** Downloaded clips land on the
phone; AirDrop them to the Mac, then `uv run tsplat ingest --src ~/Downloads`. The indexer groups files
named like Tesla's `YYYY-MM-DD_HH-MM-SS-<camera>.mp4`; if the app saves them under other names,
rename them to that pattern first. This is one clip at a time, so
it's fine for a single drive but slow for "everything". Driving the app through iPhone Mirroring
just automates the same taps, and screen-recording through Mirroring gives you a compressed,
single-view copy that's much worse for 3D reconstruction.

**Hands-off option:** a Raspberry Pi running [TeslaUSB](https://github.com/marcone/teslausb) replaces
the USB stick and syncs clips to your Mac/NAS over home Wi-Fi every time you park.

Telemetry (speed, GPS, steering, Autopilot) needs firmware 2025.44.25+ on HW3/HW4. Pillar cameras are
HW4. Older footage still works, with just no HUD data.

## 2. Reconstruct

```sh
brew install colmap        # plus Brush: see "Setup"
uv run tsplat egomask      # once: masks your own car's body using >= 8 different clips
uv run tsplat prep 2026-11-20_07-20-00 --name drive1 --start 10 --dur 20 --fps 3
uv run tsplat recon drive1             # SfM (~5-20 min CPU) + Brush training (~10-20 min on M4)
uv run tsplat view drive1              # http://127.0.0.1:8765
```

Tips:
- Pick a 15-30 s stretch at 10-40 mph with buildings, trees and signs. Highways and parked footage reconstruct poorly.
- `--fps 3` and 20 s gives 360 images. Go higher for slower driving.
- `recon --live` opens Brush's live training window. `--steps 30000` gives the best quality.
- `prep --mask-movers` (after `uv sync --extra movers`) removes cars and people from the static
  scene, so passing traffic doesn't leave smears.

## How it works

| Stage | What happens |
|---|---|
| `ingest` | rsync TeslaCam → `data/teslacam`, group per-camera files into clips (`index.json`) |
| `prep` | sample frames from all 6 cameras, mask the ego car (+ optional movers), write a rig-aware match list (same camera over time + overlapping neighbours: front↔pillars↔repeaters↔back) and an approximate rig layout, decode SEI telemetry, cut 640 px clips for the side panel |
| `recon` | COLMAP SIFT → rig configurator (6 cameras rigidly tied, layout refined by bundle adjustment) → mapper → undistort → Brush trains the splat → trajectory + metres/unit (from SEI speed) for the chase cam |
| `view` | Spark renders the splat. The chase/driver/free cameras follow the reconstructed trajectory; side videos stay synced to the same clock |

## Moving cars ("dynamic splats")

The tweet's "movers" are a 4D step: vehicles get their own Gaussians that move per frame. That needs
CUDA. The practical route is [drivestudio / OmniRe](https://github.com/ziyc/drivestudio) on a rented GPU
(one A100/4090 for an hour), fed with this pipeline's `work/<name>/dense` poses plus tracked boxes. On
the Mac you get the static world, with movers masked out so the scene stays clean.

## Image-model pass (ViewNeXt, when released)

The tweet's view is "3D scene + image model". [ViewNeXt](https://phai-lab.github.io/ViewNext/) is a video
diffusion model that renders camera trajectories from a few posed images. Its code and weights are
marked "coming soon" (checked 2026-11-22), and it is a video diffusion model, so expect it to need an
NVIDIA GPU. To have its inputs ready:

```sh
uv run tsplat nvs-export drive1   # -> work/drive1/nvs/views.json
```

`views.json` holds every registered dashcam frame with intrinsics and camera-to-world pose (OpenCV
convention), plus the chase-camera target trajectory (960×540, 55° FOV, 9 m behind and 4.2 m above
the car, 10 fps). When the model ships, convert this to its input format and use its dense-video mode.

## Setup

```sh
brew install colmap ffmpeg
mkdir -p ~/.local/opt/brush && cd ~/.local/opt/brush
curl -LO https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-aarch64-apple-darwin.tar.xz
tar xf brush-app-aarch64-apple-darwin.tar.xz     # or set BRUSH=/path/to/brush_app
uv sync
```

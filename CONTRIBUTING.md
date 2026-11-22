# Contributing

Thanks for helping. Bug reports, fixes and new features are all welcome. For larger changes,
open an issue first so we can agree on the approach.

## Dev setup (macOS, Apple Silicon)

```sh
brew install uv ffmpeg colmap   # COLMAP 3.12+ (rig support)
git clone https://github.com/jchacker5/open-tesla-reality-view && cd open-tesla-reality-view
uv sync --group dev
```

For splat training, download a [Brush](https://github.com/ArthurBrussee/brush/releases) release
binary for your platform and either put it at
`~/.local/opt/brush/brush-app-aarch64-apple-darwin/brush_app` or set `BRUSH=/path/to/brush_app`.

## Tests and lint

```sh
uv run pytest -q              # unit tests: no ffmpeg/COLMAP/Brush or footage needed
uv run ruff check .           # lint (add --fix for auto-fixes)
uv run ruff format .          # format
```

CI runs the same three commands on Ubuntu and macOS. Keep tests synthetic: build tiny fixtures
in `tmp_path` and don't use real clips.

## Footage privacy

**Never commit real dashcam footage, frames, telemetry, splats (`.ply`) or `work/` / `data/`
output.** They contain GPS traces, licence plates, faces and your home location. If you need to
share footage for a bug report or demo, strip the SEI telemetry (re-encode with ffmpeg),
blur plates and faces, and crop anything that identifies a location. `.gitignore` excludes the
usual paths, but check `git status` before you commit.

## Where we'd love help

- **Linux / CUDA training path**: a [gsplat](https://github.com/nerfstudio-project/gsplat)
  backend alongside Brush.
- **Dynamic movers**: model moving cars and people with
  [drivestudio / OmniRe](https://github.com/ziyc/drivestudio) instead of masking them out.
- **Windows support**: paths, USB drive detection, tooling install.
- **Better ego masks**: more accurate car-body masks, especially for the repeaters.
- **HW3 camera layouts**: rig priors and pairing for cars without pillar cameras.
- **Viewer features**: playback controls, measurement, export, mobile.
- **ViewNeXt integration**: wire up `tsplat nvs-export` output once the weights are released.

## Pull requests

- Keep PRs focused, and add or update tests for behaviour changes.
- `ruff check`, `ruff format --check` and `pytest` must pass.
- Describe how you tested it, and say whether you tested on real footage (don't attach the footage).

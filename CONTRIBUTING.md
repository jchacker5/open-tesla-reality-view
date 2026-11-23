# Contributing

Thanks for helping. Bug reports, fixes and new features are all welcome. For larger changes,
open an issue first so we can agree on the approach.

## Dev setup (macOS, Apple Silicon)

```sh
brew install uv ffmpeg colmap   # COLMAP 4.0+ (rig support)
git clone https://github.com/jchacker5/open-tesla-reality-view && cd open-tesla-reality-view
uv sync --group dev
uv run otrv setup-brush         # Brush v0.3.0, checksum-verified, into ~/.local/opt/brush
uv run otrv doctor              # checks every tool and prints the fix for anything missing
```

To use a Brush binary you built or downloaded yourself, set `BRUSH=/path/to/brush_app`. Use the
`otrv` command everywhere (`tsplat` is only kept as an old alias).

## Tests and lint

```sh
uv run pytest -q              # unit tests: no ffmpeg/COLMAP/Brush or footage needed
uv run ruff check .           # lint (add --fix for auto-fixes)
uv run ruff format .          # format
```

CI runs the same three commands (after `uv sync --locked --group dev`) on Python 3.11 and 3.13 on
Ubuntu and 3.12 on macOS, plus an integration job that generates synthetic 4-camera clips with
ffmpeg and runs `otrv ingest`, `otrv list` and `otrv prep` end to end. Keep tests synthetic: build
tiny fixtures in `tmp_path` and don't use real clips. External tools are faked: Brush through the
`BRUSH` env var pointing at a tiny script (see `tests/test_recon.py`), `which`/`subprocess` with
`monkeypatch` (see `tests/test_doctor.py`), and no test touches the network.

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
- **ViewNeXt integration**: wire up `otrv nvs-export` output once the weights are released.

## Pull requests

- Keep PRs focused, and add or update tests for behaviour changes.
- `ruff check`, `ruff format --check` and `pytest` must pass.
- Describe how you tested it, and say whether you tested on real footage (don't attach the footage).

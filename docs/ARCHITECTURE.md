# Architecture

How Open Tesla Reality View is put together, why it is built this way, and how changes are
tested and shipped. For usage, see the [README](../README.md).

## Goals and non-goals

| Goals | Non-goals (today) |
|---|---|
| Rebuild a recorded drive as a 3D Gaussian splat you can replay and drive through | A live, in-car visualisation |
| Run entirely on the owner's machine; footage never leaves it | Any cloud service, account or telemetry |
| Work with the files the car already writes (`TeslaCam/`), 4- or 6-camera | Requiring calibration targets or extra sensors |
| Every stage re-runnable on its own, with outputs on disk you can inspect | One opaque end-to-end command |
| Measurable quality, so changes are judged by numbers ([HILL_CLIMB.md](HILL_CLIMB.md)) | Dynamic actors (moving cars are masked, not rebuilt) |

## System overview

```
            ┌──────────── otrv CLI (tsplat/__main__.py) ─────────────┐    otrv studio (FastAPI, 127.0.0.1)
            │                                                        │    queues the same CLI as subprocesses
TeslaCam/ ─▶ ingest ─▶ prep ─▶ recon ─▶ clean ─▶ collision ─▶ bench    │
 (USB)       │         │        │                               │    │
             ▼         ▼        ▼                               ▼    │
          index.json  frames   COLMAP SfM (rig) ─▶ undistort    ledger.jsonl
          + speed     masks    ─▶ Brush training (Metal)             │
                      rig prior        │                             │
                      telemetry        ▼                             │
                                  viewer/scene.ply, path.json ─▶ view (static server, Range)
                                                                 render (headless Chromium ─▶ MP4)
                                  C++ core (cpp/): SEI decode, GPS georeferencing ─▶ georef.json
```

The pipeline is a chain of **idempotent stages that communicate only through files** in
`work/<name>/`. Each stage validates its inputs, writes its outputs, and can be re-run alone
(`otrv clean`, `otrv collision` and `otrv bench` re-run without retraining). The studio web app and
CI both drive the same CLI, so there is one code path to test.

## Components

| Area | Modules | Responsibility |
|---|---|---|
| CLI | `tsplat/__main__.py` | Argument parsing, friendly errors, dispatch to stages |
| Ingest | `ingest.py`, `speed.py`, `sei.py` | rsync from the USB drive, group per-camera files into clips, cache per-clip speed |
| Telemetry | `sei.py`, `cpp/src/sei.cpp` | Decode SEI protobuf metadata (speed, gear, steering, Autopilot, GPS) from H.264 |
| Dataset prep | `prep.py`, `sky.py` | Sample frames, ego/mover/sky masks, rig prior, match pairs, viewer side-panel clips |
| Reconstruction | `recon.py`, `brush.py`, `densify.py`, `colornorm.py`, `holdout.py` | COLMAP features → rig configurator → mapper → undistort → Brush training |
| Post-processing | `clean.py`, `collision.py`, `splats.py`, `georef.py` | Prune and dehaze the splat, bake drive-mode collision, fit to GPS |
| Evaluation | `bench.py`, `holdout.py` | Registration, reprojection, held-out PSNR/LPIPS, timings; gate and log every run |
| Output | `serve.py`, `render.py`, `enhance.py`, `viewer/` | Local replay viewer, cinematic MP4 render, optional Difix clean-up |
| Studio | `tsplat/studio/` | Local web app: sources, window picking, single-GPU FIFO build queue, live progress |
| Native core | `cpp/` | C++20 SEI decoder and georeferencing, nanobind bindings, CLI (`otrv-sei`) |
| Tooling | `doctor.py`, `brush.py` | Environment checks with exact fixes; checksum-verified Brush install |

## Data contract: `work/<name>/`

| Path | Written by | Read by |
|---|---|---|
| `meta.json` | prep | recon, bench, studio |
| `images/<cam>/`, `masks/<cam>/` | prep | recon (COLMAP, Brush) |
| `rig_config.json`, `pairs.txt` | prep | recon (rig configurator, matcher) |
| `sparse/`, `dense/` | recon | clean, bench, georef |
| `viewer/scene.ply`, `viewer/scene_raw.ply` | recon / clean | view, render, collision |
| `viewer/path.json`, `viewer/telemetry.json`, `viewer/cams/` | prep / recon | view, render |
| `viewer/collision.json` | collision | viewer drive mode |
| `viewer/georef.json` | georef | viewer map |
| `metrics.json`, `timings.json`, `recon_args.json` | recon / bench | bench, studio |
| `work/ledger.jsonl` | bench | hill-climb loop |

Changing a file's shape is a breaking change for every reader in the right-hand column. Those
changes need a test on both sides and a CHANGELOG entry.

## Key design decisions

1. **Solve all cameras as one rigid rig.** Independent per-camera SfM splits into disconnected
   models on low-overlap Tesla layouts. Seeding COLMAP's rig configurator with an approximate
   Tesla camera layout, then letting bundle adjustment refine it, keeps every camera in one
   metric model.
2. **Metric scale from the car, not the images.** SfM is scale-free. Telemetry speed integrated
   over time gives real-world scale, and `georef` cross-checks it against GPS (true north and a
   map origin).
3. **Two telemetry decoders, one behaviour.** The pure-Python decoder is the readable reference
   and the fallback. The C++ port is about 3x faster and matches it rule for rule. Thousands of
   fuzzed inputs and a real clip in CI prove the two agree. Every length in the file is treated as
   hostile; see [cpp/README.md](../cpp/README.md).
4. **Local-first, private by default.** The servers bind `127.0.0.1`, and the studio checks
   `Host`/`Origin` and needs a token. Map tiles are opt-in because they reveal location. Footage,
   splats and `work/` are git-ignored and must never be committed.
5. **Measure every change.** `otrv bench` scores and gates each reconstruction and appends to a
   ledger, so a quality change is a number, not a screenshot ([HILL_CLIMB.md](HILL_CLIMB.md)).
6. **Shell out to best-in-class tools.** COLMAP (SfM), Brush (Metal splat training), ffmpeg and
   Chromium are invoked as subprocesses behind `doctor`-checked, version-pinned interfaces, not
   reimplemented.

## Quality and testing strategy

| Layer | What | Where |
|---|---|---|
| Static checks | `ruff check`, `ruff format --check`; `-Werror` C++ builds; `.clang-format` | CI `test`, `native` |
| Unit (Python) | ~270 tests, fully synthetic: tiny fixtures in `tmp_path`, external tools faked, no network | `tests/`, CI `test` on Linux + macOS, Python 3.11-3.13 |
| Unit (C++) | doctest suites, including 20,000 randomized inputs, under ASan + UBSan | `cpp/tests/`, CI `native` on Linux + macOS |
| Parity | C++ vs Python decoder on ~7,000 fuzzed protobufs/MP4s and a real H.264 clip, byte-for-byte JSON | `tests/test_sei_native.py` |
| Integration | Synthetic 4-camera TeslaCam (ffmpeg `testsrc2`) through `ingest → list → prep`, wheel contents, error messages | CI `integration` |
| Render | Real capture loop in headless Chromium on SwiftShader (CPU WebGL), ffmpeg encode | CI `render` |
| Quality | Held-out PSNR/LPIPS, registration and reprojection gates on real footage | `otrv bench` (local; footage can't go in CI) |
| Security | CodeQL (Python, JS, C/C++), Dependabot, private vulnerability reporting | `.github/`, [SECURITY.md](../SECURITY.md) |

## Development lifecycle

1. **Plan.** Work starts from an issue (bug or feature template). Larger changes agree on the
   approach in the issue first.
2. **Branch.** Short-lived `feat/…`, `fix/…`, `docs/…` or `chore/…` branches off `main`.
3. **Build and test locally.** `uv run pytest -q`, `ruff`, and for `cpp/` a sanitizer build
   ([CONTRIBUTING.md](../CONTRIBUTING.md)). Pipeline changes are also run on real footage and
   scored with `otrv bench`.
4. **Pull request.** The template asks for what/why, how it was tested, and a privacy check.
   `main` is protected: CI must pass before merge, and branches are squash-merged.
5. **Release.** Semantic versioning. Update `CHANGELOG.md` and the version in `pyproject.toml`,
   then push a `vX.Y.Z` tag. The release workflow builds the wheel and sdist and publishes a
   GitHub Release with the changelog section.
6. **Operate.** Bugs come in through issues and vulnerabilities through private advisories.
   Dependabot keeps dependencies and Actions current, and CodeQL scans every PR and runs weekly.

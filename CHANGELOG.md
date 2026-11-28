# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- `otrv render` on a fresh reconstruction stopped with "car.js is missing": it now installs the
  viewer modules, car model and `features.json` the same way `otrv view` does.
- `otrv render --enhance` failed at model load after a Dependabot bump of `peft` (0.9.0 -> 0.22.0,
  which needs transformers 5.6+). The Difix pins are restored, Dependabot ignores them, and a test
  guards them.
- `otrv recon` stopped at random with Brush's "IO error while loading dataset: early eof", even
  though the same dataset loads fine on the next try. It now retries that load error (up to 3
  times). Any other Brush error still stops the run straight away.
- With a source-built Brush 1.x (`otrv setup-brush --from-source`), `otrv recon` trained and then
  reported "Brush exported nothing". Brush 1.x resolves a relative `--export-path` against the
  dataset's parent folder, so the splat landed in `work/<name>/work/<name>/splat`. recon now passes
  an absolute path.

## [0.1.0] - 2026-11-23

First public release.

### Added
- **Pipeline:** TeslaCam ingest, frame and mask prep, rig-aware COLMAP SfM, Brush splat training
  on Metal, and a replay viewer. Supports 4- and 6-camera cars (#1).
- **Telemetry:** SEI decoder (speed, gear, steering, Autopilot, GPS) in pure Python.
- **Evaluation:** `otrv bench` scores, gates and logs every reconstruction for hill-climbing (#6).
- **Viewer v2:** drive mode, real car model support, cinematic camera, HUD, render API and an
  opt-in street map (#10, #15).
- **Studio:** a local web app to pick footage, queue builds with live progress and open results
  (#14).
- **Pipeline v2:** bench v2, post-train clean-up, transparent sky, dense depth init, colour
  normalisation, 6 fps sampling, Brush from source and a collision bake (#16).
- **Render:** `otrv render` makes cinematic MP4s, with optional NVIDIA Difix clean-up (#12).
- **C++ core:** a C++20 SEI decoder with nanobind Python bindings, checked against the Python
  decoder on fuzzed input under ASan/UBSan (#18).
- **Georeferencing:** a GPS fit gives metric scale, true north and a map origin (#19).
- `doctor`, `setup-brush` and `suggest` commands, plus friendlier CLI errors (#11).
- **Project:** an architecture and lifecycle guide ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)),
  CodeQL scanning (Python, JavaScript, C/C++), Dependabot, CODEOWNERS, and a tag-driven release
  workflow.

### Fixed
- Issues found in the first run on real footage (#9).
- Ingest ignores macOS `._*` AppleDouble files (#8).

[Unreleased]: https://github.com/jchacker5/open-tesla-reality-view/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jchacker5/open-tesla-reality-view/releases/tag/v0.1.0

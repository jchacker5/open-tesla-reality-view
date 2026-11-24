# otrv C++ core

The parts of the pipeline that are worth having in C++, as a small C++20 library with no
third-party runtime dependencies, a command-line tool, and Python bindings that `tsplat` picks up
automatically.

| Component | What it does | Python counterpart |
|---|---|---|
| `otrv/sei.hpp` | Tesla dashcam SEI telemetry: MP4 atom walk, AVCC NAL parsing, H.264 emulation-prevention removal, a protobuf wire-format decoder for `SeiMetadata` | `tsplat/sei.py` |
| `otrv/geo.hpp` | Georeferencing: WGS84 / ECEF / ENU conversions and a robust fit of the reconstructed path to the car's GPS (metric scale, true north, map origin) | (new; `tsplat/georef.py` does the file I/O) |
| `otrv-sei` | CLI: a clip's telemetry as JSON (byte-identical to `python -m tsplat.sei`), or `--summary` | `python -m tsplat.sei` |
| `otrv_native` | nanobind module; `tsplat.sei.extract` uses it when installed | |

## Build

```sh
# library, CLI and tests (CMake 3.20+, any C++20 compiler)
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/otrv-sei --summary data/teslacam/RecentClips/2026-11-20_07-20-00-front.mp4

# Python extension (scikit-build-core + nanobind), installed into the project's venv
uv sync --extra native
```

Options: `-DOTRV_SANITIZE=ON` (ASan + UBSan), `-DOTRV_WERROR=ON`. CI runs both on Linux and macOS.

## Design notes

**One decoder, two implementations.** The Python decoder stays as the readable reference and the
fallback when the extension isn't built. The C++ port follows it rule for rule, malformed input
included: a truncated fixed-width field fails the message, a length-delimited field whose length
runs past the end ends it, a varint over 64 bits is rejected, unknown fields are skipped, `bool`
fields coerce whatever wire type they arrive in. Fields keep wire order, so even the dict key
order matches. `tests/test_sei_native.py` checks this on ~7,000 fuzzed protobufs and MP4s (random
fields and wire types, truncation, flipped header bytes) and on a real H.264 clip, and the CLI's
JSON is compared byte for byte with `json.dumps(rows, indent=1)`. `OTRV_NATIVE=0` forces the
Python path.

**Untrusted input.** Every length in the file (atom sizes, NAL length prefixes, protobuf lengths
and varints) is treated as hostile: reads are bounds-checked, buffers are sized by what's left in
the file rather than by the length prefix, and sizes saturate instead of overflowing. The unit
tests include 20,000 randomized inputs and run under AddressSanitizer and UndefinedBehaviorSanitizer.

**I/O, not parsing, is the cost.** A minute of 1280x960 front-camera video is ~76 MB, of which the
telemetry is a few hundred bytes per frame. `FileSource` reads through a 4 KiB window: one read
covers a NAL length prefix, the SEI message and the next slice header, and the video slices in
between are skipped. (4 KiB beat 16 and 64 KiB by 3-4x on a full-resolution clip, because larger
windows end up reading the slices too.) The GIL is released while decoding.

Measured on an M4 Mac, warm page cache, best of 9, identical output:

| Clip | Python | C++ via `otrv_native` |
|---|---|---|
| 60 s, 1280x960, 36 fps (2,160 frames, 76 MB) | 22.5 ms | 7.8 ms |
| 60 s, 480x360, 24 fps (1,440 frames, 11 MB) | 15.1 ms | 4.9 ms |

The C++ time includes building the Python dicts; the decode alone is 6.7 ms on the large clip.

## Georeferencing with the car's GPS

SfM recovers a drive only up to a similarity: its scale, orientation and position are arbitrary.
The pipeline already sets scale from the speedometer (integrated SEI speed over the path length).
`otrv/geo.hpp` adds the GPS, which is in every SEI frame too: fitting the reconstructed path to the
GPS track gives metric scale, true north and a geographic origin in one solve, and a second,
independent scale estimate to check the first against.

- **Model.** SEI has latitude and longitude but no altitude, and a car's attitude is mostly yaw. So
  the scene's up vector (from the rig) is rotated onto local vertical first, and then a 2D
  similarity (scale, yaw, east/north offset, 4 DoF) is fitted in the horizontal plane. A full
  7-DoF Umeyama fit would be ill-posed on a straight road, where roll about the road axis is
  unobservable from the track. The 2D least-squares solution is closed form in complex arithmetic,
  `z = Σ conj(a_i - ā)(b_i - b̄) / Σ |a_i - ā|²`, so no SVD or linear-algebra dependency is needed.
- **Robustness.** RANSAC with 2-point minimal samples (exhaustive when there are few enough pairs,
  sampled otherwise), then least squares on the inliers, repeated until the inlier set stops
  changing. Pairs that are too close together to condition the fit are skipped. A fit is rejected,
  with a reason, if the car barely moved, the path and GPS don't overlap in time, or under half the
  path agrees with the GPS.
- **Tesla's GPS stream.** The car writes the latest fix into every frame (36 fps) but the receiver
  updates far less often, so a run of identical fixes counts once, at its first frame. `(0, 0)` is
  "no fix". Positions are interpolated linearly between fixes, never extrapolated, and dropouts
  longer than 2 s (tunnels, garages) aren't bridged.
- **Geodesy.** Exact WGS84 geodetic/ECEF conversion (Bowring with fixed-point refinement,
  sub-millimetre) and a local ENU tangent frame at the first fix.

`otrv georef <run>` (and `otrv recon`, when the extension is installed) writes
`viewer/georef.json`: the transform as a 4x4 matrix, metres per unit, heading offset, the origin,
inlier count and RMS residual, every path sample's latitude/longitude, and the ratio of the GPS
scale to the speed scale. The two scales come from different sensors, so a ratio far from 1 points
at a bad reconstruction or bad GPS; `otrv georef` says so when they differ by more than 10%.

Tests: synthetic drives (a straight road and a 90° turn) under an arbitrary similarity, with
10 Hz GPS, 1 m receiver noise and 10% multipath jumps. Scale comes back within 2% and every
position within 2 m, the jumps are rejected, and each rejection path is exercised. Not yet
run on real footage: the clips on this machine have no SEI telemetry, so the first real check is
`otrv georef` on a drive recorded with firmware 2025.44.25+.

## Layout

```
cpp/
  include/otrv/   public headers
  src/            library
  tools/          command-line tools
  python/         nanobind bindings + the otrv_native package
  tests/          doctest unit tests (doctest is downloaded at configure time, pinned by SHA-256)
```

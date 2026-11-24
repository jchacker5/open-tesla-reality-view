# otrv C++ core

The parts of the pipeline that are worth having in C++, as a small C++20 library with no
third-party runtime dependencies, a command-line tool, and Python bindings that `tsplat` picks up
automatically.

| Component | What it does | Python counterpart |
|---|---|---|
| `otrv/sei.hpp` | Tesla dashcam SEI telemetry: MP4 atom walk, AVCC NAL parsing, H.264 emulation-prevention removal, a protobuf wire-format decoder for `SeiMetadata` | `tsplat/sei.py` |
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

## Layout

```
cpp/
  include/otrv/   public headers
  src/            library
  tools/          command-line tools
  python/         nanobind bindings + the otrv_native package
  tests/          doctest unit tests (doctest is downloaded at configure time, pinned by SHA-256)
```

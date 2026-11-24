"""Decode Tesla dashcam SEI telemetry (speed, GPS, heading, steering...) from an MP4.

Port of teslamotors/dashcam sei_extractor.py with a tiny built-in protobuf decoder,
so no protoc / protobuf dependency is needed. Field numbers follow dashcam.proto.
Requires firmware 2025.44.25+ on HW3/HW4; parked clips may have no SEI.

`extract` uses the C++ decoder in `otrv_native` (cpp/, `uv sync --extra native`) when it is
installed and falls back to this pure-Python one otherwise. Both apply the same rules to the same
bytes; tests/test_sei_native.py checks that on fuzzed input. Set OTRV_NATIVE=0 to force Python.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

# field number -> (name, wire kind)
FIELDS = {
    1: ("version", "varint"),
    2: ("gear_state", "varint"),
    3: ("frame_seq_no", "varint"),
    4: ("vehicle_speed_mps", "f32"),
    5: ("accelerator_pedal_position", "f32"),
    6: ("steering_wheel_angle", "f32"),
    7: ("blinker_on_left", "bool"),
    8: ("blinker_on_right", "bool"),
    9: ("brake_applied", "bool"),
    10: ("autopilot_state", "varint"),
    11: ("latitude_deg", "f64"),
    12: ("longitude_deg", "f64"),
    13: ("heading_deg", "f64"),
    14: ("linear_acceleration_mps2_x", "f64"),
    15: ("linear_acceleration_mps2_y", "f64"),
    16: ("linear_acceleration_mps2_z", "f64"),
}
GEARS = ["P", "D", "R", "N"]
AUTOPILOT = ["NONE", "SELF_DRIVING", "AUTOSTEER", "TACC"]


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            if result >> 64:
                raise ValueError("varint exceeds 64 bits")
            return result, i
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def decode_proto(buf: bytes) -> dict | None:
    out: dict = {}
    i = 0
    try:
        while i < len(buf):
            key, i = _varint(buf, i)
            num, wt = key >> 3, key & 7
            if wt == 0:
                val, i = _varint(buf, i)
            elif wt == 1:
                val = struct.unpack_from("<d", buf, i)[0]
                i += 8
            elif wt == 5:
                val = struct.unpack_from("<f", buf, i)[0]
                i += 4
            elif wt == 2:
                ln, i = _varint(buf, i)
                i += ln
                continue
            else:
                return None
            if num in FIELDS:
                name, kind = FIELDS[num]
                out[name] = bool(val) if kind == "bool" else val
    except (IndexError, struct.error, ValueError):
        return None
    return out or None


def _strip_epb(data: bytes) -> bytes:
    out = bytearray()
    zeros = 0
    for b in data:
        if zeros >= 2 and b == 3:
            zeros = 0
            continue
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def _payload(nal: bytes) -> bytes | None:
    for i in range(3, len(nal) - 1):
        b = nal[i]
        if b == 0x42:
            continue
        if b == 0x69:
            return _strip_epb(nal[i + 1 : -1]) if i > 2 else None
        break
    return None


def _find_mdat(fp) -> tuple[int, int]:
    fp.seek(0)
    while True:
        hdr = fp.read(8)
        if len(hdr) < 8:
            raise RuntimeError("mdat atom not found")
        size32, typ = struct.unpack(">I4s", hdr)
        if size32 == 1:
            ext = fp.read(8)
            if len(ext) < 8:
                raise RuntimeError("truncated MP4 atom header")
            size, hsize = struct.unpack(">Q", ext)[0], 16
        else:
            size, hsize = size32, 8
        if typ == b"mdat":
            return fp.tell(), (size - hsize if size else 0)
        if size < hsize:
            raise RuntimeError("invalid MP4 atom size")
        fp.seek(size - hsize, 1)


def _label(table: list[str], value) -> str:
    """Enum name for a decoded value; "?" if it is out of range or not an integer (a missing field is 0)."""
    return table[value] if isinstance(value, int) and 0 <= value < len(table) else "?"


try:
    from otrv_native import sei_extract as _native_extract
except ImportError:  # the C++ extension is optional
    _native_extract = None


def native_available() -> bool:
    return _native_extract is not None


def extract(path: str | Path) -> list[dict]:
    """One dict per SEI telemetry message in the clip, in file order."""
    if _native_extract is not None and os.environ.get("OTRV_NATIVE", "1") != "0":
        return _native_extract(str(path))
    return extract_py(path)


def extract_py(path: str | Path) -> list[dict]:
    """Pure-Python reference decoder."""
    rows = []
    with open(path, "rb") as fp:
        off, size = _find_mdat(fp)
        fp.seek(off)
        used = 0
        while size == 0 or used < size:
            hdr = fp.read(4)
            if len(hdr) < 4:
                break
            n = struct.unpack(">I", hdr)[0]
            if n < 2:
                fp.seek(n, 1)
                used += 4 + n
                continue
            two = fp.read(2)
            if len(two) != 2:
                break
            used += 4 + n
            if (two[0] & 0x1F) != 6 or two[1] != 5:  # SEI / user_data_unregistered
                fp.seek(n - 2, 1)
                continue
            rest = fp.read(n - 2)
            p = _payload(two + rest)
            meta = decode_proto(p) if p else None
            if meta and "frame_seq_no" in meta:
                meta["gear"] = _label(GEARS, meta.get("gear_state", 0))
                meta["autopilot"] = _label(AUTOPILOT, meta.get("autopilot_state", 0))
                rows.append(meta)
    return rows


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m tsplat.sei <clip.mp4>")
    rows = extract(sys.argv[1])
    if not rows:
        sys.exit("No SEI metadata (needs firmware 2025.44.25+, HW3+, car not parked)")
    json.dump(rows, sys.stdout, indent=1)

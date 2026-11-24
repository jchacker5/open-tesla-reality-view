"""The C++ decoder (otrv_native, built from cpp/) must agree with the Python one byte for byte.

Skipped when the extension isn't installed (`uv sync --extra native`); CI sets OTRV_REQUIRE_NATIVE=1
so a broken build fails instead of skipping. OTRV_PARITY_CLIP=/path/to/clip-front.mp4 adds a real clip.
"""

import json
import os
import random
import struct
import subprocess

import pytest
from test_sei import add_epb, key, mp4, proto, sei_nal, varint

from tsplat import sei

if not sei.native_available():
    if os.environ.get("OTRV_REQUIRE_NATIVE") == "1":
        raise RuntimeError("OTRV_REQUIRE_NATIVE=1 but otrv_native is not importable")
    pytest.skip("otrv_native not installed (uv sync --extra native)", allow_module_level=True)

import otrv_native  # noqa: E402


def canon(rows):
    """Order- and type-exact form: key order, int vs float vs bool, NaN and -0.0 all count."""
    return repr([list(r.items()) for r in rows]) if rows is not None else None


def py_extract(path):
    try:
        return "ok", canon(sei.extract_py(path))
    except RuntimeError:
        return "RuntimeError", None


def native_extract(path):
    try:
        return "ok", canon(otrv_native.sei_extract(str(path)))
    except RuntimeError:
        return "RuntimeError", None


def random_field(rng: random.Random) -> bytes:
    num = rng.choice([*range(1, 17), 0, 17, 40, 2047])
    wire = rng.choices([0, 1, 5, 2, 3, 4, 6, 7], weights=[8, 4, 4, 2, 1, 1, 1, 1])[0]
    body = key(num, wire)
    if wire == 0:
        body += varint(rng.choice([0, 1, 2, 3, 4, 5, rng.getrandbits(7), rng.getrandbits(35), rng.getrandbits(64)]))
    elif wire == 1:
        special = [0.0, -0.0, 1.5, 37.7749, float("nan"), float("inf")]
        body += struct.pack("<d", rng.choice([*special, rng.uniform(-1e9, 1e9)]))
    elif wire == 5:
        body += struct.pack("<f", rng.choice([0.0, 12.3, -1.0, rng.uniform(-100, 100)]))
    elif wire == 2:
        n = rng.randrange(6)
        body += varint(n if rng.random() < 0.9 else 500) + rng.randbytes(n)
    return body


def random_proto(rng: random.Random) -> bytes:
    p = b"".join(random_field(rng) for _ in range(rng.randrange(1, 12)))
    if rng.random() < 0.7:
        p = key(3, 0) + varint(rng.getrandbits(20)) + p  # most real messages carry frame_seq_no
    if rng.random() < 0.15:
        p = p[: rng.randrange(len(p) + 1)]  # truncated
    return p


def random_clip(rng: random.Random) -> bytes:
    nals = []
    for _ in range(rng.randrange(1, 8)):
        r = rng.random()
        if r < 0.6:
            nals.append(sei_nal(random_proto(rng)))
        elif r < 0.75:
            nals.append(bytes([0x65, 0x88]) + rng.randbytes(rng.randrange(20)))
        elif r < 0.85:
            nals.append(bytes([0x06, 0x05]) + rng.randbytes(rng.randrange(30)))  # SEI with a non-Tesla UUID
        else:
            nals.append(rng.randbytes(rng.randrange(3)))  # 0-2 byte NALs
    data = bytearray(mp4(*nals))
    for _ in range(rng.choice([0, 0, 0, 1, 3])):  # flip bytes, including in the atom headers
        data[rng.randrange(len(data))] = rng.randrange(256)
    if rng.random() < 0.1:
        data = data[: rng.randrange(len(data) + 1)]
    return bytes(data)


def test_fixture_matches(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(mp4(sei_nal(proto(gear=1, seq=100, speed=12.5, brake=True, lat=37.7749, lon=-122.4194))))
    assert native_extract(f) == py_extract(f)
    assert otrv_native.sei_extract(str(f))[0]["gear"] == "D"


def test_dispatch_uses_native_and_can_be_disabled(tmp_path, monkeypatch):
    f = tmp_path / "clip.mp4"
    f.write_bytes(mp4(sei_nal(proto(gear=2, seq=5, speed=1.0, brake=False, lat=0.0, lon=0.0))))
    calls = []
    monkeypatch.setattr(sei, "_native_extract", lambda p: calls.append(p) or [])
    assert sei.extract(f) == [] and calls == [str(f)]
    monkeypatch.setenv("OTRV_NATIVE", "0")
    assert sei.extract(f)[0]["frame_seq_no"] == 5 and len(calls) == 1


@pytest.mark.parametrize("seed", range(4))
def test_fuzzed_clips_match(tmp_path, seed):
    rng = random.Random(seed)
    f = tmp_path / "clip.mp4"
    for i in range(500):
        f.write_bytes(random_clip(rng))
        assert native_extract(f) == py_extract(f), f"seed {seed} case {i}"


def test_fuzzed_protos_match():
    rng = random.Random(99)
    for i in range(5000):
        p = random_proto(rng) if i % 2 else rng.randbytes(rng.randrange(40))
        want = sei.decode_proto(p)
        got = otrv_native.sei_decode_proto(p)
        assert canon([got] if got else None) == canon([want] if want else None), p.hex()


def test_epb_heavy_payload_matches(tmp_path):
    p = key(3, 0) + varint(1) + b"".join(key(14, 1) + struct.pack("<d", 0.0) for _ in range(50))
    assert add_epb(p) != p
    f = tmp_path / "clip.mp4"
    f.write_bytes(mp4(sei_nal(p)))
    assert native_extract(f) == py_extract(f)


def test_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        otrv_native.sei_extract(str(tmp_path / "nope.mp4"))


@pytest.mark.skipif(not os.environ.get("OTRV_PARITY_CLIP"), reason="set OTRV_PARITY_CLIP to a real TeslaCam clip")
def test_real_clip_matches():
    path = os.environ["OTRV_PARITY_CLIP"]
    rows = sei.extract_py(path)
    assert rows, "clip has no SEI telemetry"
    assert native_extract(path) == ("ok", canon(rows))


@pytest.mark.skipif(not os.environ.get("OTRV_SEI_BIN"), reason="set OTRV_SEI_BIN to the built otrv-sei tool")
def test_cli_json_matches_python_json(tmp_path):
    rng = random.Random(7)
    f = tmp_path / "clip.mp4"
    f.write_bytes(mp4(*(sei_nal(key(3, 0) + varint(i) + random_proto(rng)) for i in range(200))))
    rows = sei.extract_py(f)
    assert rows
    out = subprocess.run([os.environ["OTRV_SEI_BIN"], str(f)], capture_output=True, text=True, check=True).stdout
    assert out == json.dumps(rows, indent=1)

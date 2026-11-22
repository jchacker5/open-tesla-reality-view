import struct

import pytest

from tsplat import sei


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def key(num: int, wire: int) -> bytes:
    return varint(num << 3 | wire)


def proto(**f) -> bytes:
    return b"".join(
        [
            key(2, 0) + varint(f["gear"]),
            key(3, 0) + varint(f["seq"]),
            key(4, 5) + struct.pack("<f", f["speed"]),
            key(9, 0) + varint(int(f["brake"])),
            key(11, 1) + struct.pack("<d", f["lat"]),
            key(12, 1) + struct.pack("<d", f["lon"]),
            key(13, 1) + struct.pack("<d", 0.0),  # eight zero bytes -> needs emulation prevention
        ]
    )


def add_epb(data: bytes) -> bytes:
    """H.264 emulation prevention: insert 0x03 after 00 00 when the next byte is <= 3."""
    out, zeros = bytearray(), 0
    for b in data:
        if zeros >= 2 and b <= 3:
            out.append(3)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def sei_nal(payload: bytes) -> bytes:
    body = b"\x42" * 16 + b"\x69" + add_epb(payload)
    return bytes([0x06, 0x05, min(len(body), 255)]) + body + b"\x80"


def mp4(*nals: bytes) -> bytes:
    data = b"".join(struct.pack(">I", len(n)) + n for n in nals)
    ftyp = struct.pack(">I4s", 16, b"ftyp") + b"isom\x00\x00\x02\x00"
    return ftyp + struct.pack(">I4s", 8 + len(data), b"mdat") + data


def test_extract_decodes_sei(tmp_path):
    p1 = proto(gear=1, seq=100, speed=12.5, brake=True, lat=37.7749, lon=-122.4194)
    p2 = proto(gear=2, seq=101, speed=0.0, brake=False, lat=37.775, lon=-122.42)
    assert b"\x00\x00\x00" in p1 and add_epb(p1) != p1  # the fixture really exercises EPB stripping
    idr = bytes([0x65, 0x88]) + b"\x00" * 10
    other_sei = bytes([0x06, 0x04]) + b"\x01\x02\x03"  # SEI but not user_data_unregistered
    f = tmp_path / "clip.mp4"
    f.write_bytes(mp4(idr, sei_nal(p1), other_sei, sei_nal(p2)))

    rows = sei.extract(f)
    assert len(rows) == 2
    r = rows[0]
    assert r["frame_seq_no"] == 100
    assert r["gear"] == "D" and r["gear_state"] == 1
    assert r["vehicle_speed_mps"] == pytest.approx(12.5)
    assert r["brake_applied"] is True
    assert r["latitude_deg"] == 37.7749 and r["longitude_deg"] == -122.4194
    assert r["heading_deg"] == 0.0
    assert r["autopilot"] == "NONE"
    assert rows[1]["frame_seq_no"] == 101 and rows[1]["gear"] == "R" and rows[1]["brake_applied"] is False


def test_extract_without_mdat_raises(tmp_path):
    f = tmp_path / "x.mp4"
    f.write_bytes(struct.pack(">I4s", 16, b"ftyp") + b"isom\x00\x00\x02\x00")
    with pytest.raises(RuntimeError):
        sei.extract(f)


def test_strip_epb():
    assert sei._strip_epb(b"\x00\x00\x03\x01\x00\x00\x03\x00") == b"\x00\x00\x01\x00\x00\x00"
    assert sei._strip_epb(b"\x01\x03\x00\x03") == b"\x01\x03\x00\x03"


@pytest.mark.parametrize("buf", [b"", b"\xff\xff\xff", b"\x0b", b"\x25\x01\x02", b"\x1f\x00"])
def test_decode_proto_garbage(buf):
    assert sei.decode_proto(buf) is None

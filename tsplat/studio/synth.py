"""Synthetic TeslaCam footage for trying the studio without a car (and for its tests and screenshots).

    python -m tsplat.studio.synth /tmp/TeslaCam            # 3 drives: Recent, Saved (6 cams), Sentry

Each camera file is real H.264 (ffmpeg testsrc2) with a Tesla-style SEI telemetry message in every
frame (speed, GPS, heading, gear), so speed sparklines, routes and "best window" work like real footage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

CAM_HUES = {
    "front": 0,
    "back": 60,
    "left_repeater": 120,
    "right_repeater": 180,
    "left_pillar": 240,
    "right_pillar": 300,
}


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def _key(num: int, wire: int) -> bytes:
    return _varint(num << 3 | wire)


def telemetry_proto(seq: int, mps: float, lat: float, lon: float, heading: float, gear: int = 1) -> bytes:
    """A dashcam.proto SeiMetadata message (field numbers as in tsplat/sei.py)."""
    return b"".join(
        [
            _key(1, 0) + _varint(1),
            _key(2, 0) + _varint(gear),
            _key(3, 0) + _varint(seq),
            _key(4, 5) + struct.pack("<f", mps),
            _key(11, 1) + struct.pack("<d", lat),
            _key(12, 1) + struct.pack("<d", lon),
            _key(13, 1) + struct.pack("<d", heading),
        ]
    )


def _epb(data: bytes) -> bytes:
    out, zeros = bytearray(), 0
    for b in data:
        if zeros >= 2 and b <= 3:
            out.append(3)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def sei_nal(payload: bytes) -> bytes:
    """SEI NAL, user_data_unregistered, Tesla framing (0x42 padding + 0x69 marker before the protobuf)."""
    body = b"\x42" * 16 + b"\x69" + payload
    size = len(body)
    hdr = b"\x05" + b"\xff" * (size // 255) + bytes([size % 255])
    return b"\x06" + _epb(hdr + body) + b"\x80"


def _split_annexb(data: bytes) -> list[bytes]:
    out, i, n = [], 0, len(data)
    starts = []
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        starts.append(j + 3)
        i = j + 3
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else n
        nal = data[s:e]
        while nal.endswith(b"\x00"):
            nal = nal[:-1]
        out.append(nal)
    return out


def write_clip(
    path: Path, seconds: float, fps: int, size: str, hue: int, frames: list[bytes] | None, seq0: int = 0
) -> None:
    """Encode testsrc2 to H.264, insert one SEI per access unit (after each AUD), remux to MP4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "v.h264"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi"]
            + ["-i", f"testsrc2=size={size}:rate={fps}", "-t", f"{seconds}", "-vf", f"hue=h={hue}"]
            + ["-c:v", "libx264", "-preset", "ultrafast", "-g", str(fps), "-pix_fmt", "yuv420p"]
            + ["-x264-params", "aud=1", "-bsf:v", "h264_mp4toannexb", "-f", "h264", str(raw)],
            check=True,
        )
        out = bytearray()
        au = 0
        for nal in _split_annexb(raw.read_bytes()):
            out += b"\x00\x00\x00\x01" + nal
            if nal and nal[0] & 0x1F == 9 and frames is not None:  # access unit delimiter
                if au < len(frames):
                    out += b"\x00\x00\x00\x01" + sei_nal(frames[au])
                au += 1
        raw.write_bytes(bytes(out))
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-r", str(fps), "-f", "h264", "-i", str(raw)]
            + ["-c", "copy", "-movflags", "+faststart", str(path)],
            check=True,
        )


def speed_profile(t: float, base: float) -> float:
    """mph: pull away, cruise with gentle variation, slow for a light."""
    s = base + 6 * math.sin(t / 9.0) + 3 * math.sin(t / 3.7)
    if t < 8:
        s *= t / 8
    return max(0.0, s)


def make_drive(
    root: Path,
    folder: str,
    start: dt.datetime,
    n_clips: int,
    cams: list[str],
    seconds: float = 60,
    fps: int = 24,
    size: str = "480x360",
    base_mph: float = 26,
    lat0: float = 37.7793,
    lon0: float = -122.4193,
    heading0: float = 40.0,
    parked: bool = False,
    event: dict | None = None,
) -> list[str]:
    """Write `n_clips` consecutive clips of one drive into root/folder/. Returns the clip timestamps."""
    lat, lon, heading = lat0, lon0, heading0
    t_total = 0.0
    stamps = []
    for c in range(n_clips):
        ts = (start + dt.timedelta(seconds=c * seconds)).strftime("%Y-%m-%d_%H-%M-%S")
        stamps.append(ts)
        frames = []
        for f in range(int(seconds * fps)):
            t = t_total + f / fps
            mph = 0.0 if parked else speed_profile(t, base_mph)
            mps = mph / 2.23694
            heading = (heading + 0.6 * math.sin(t / 14.0) / fps * 10) % 360
            d = mps / fps
            lat += d * math.cos(math.radians(heading)) / 111_320
            lon += d * math.sin(math.radians(heading)) / (111_320 * math.cos(math.radians(lat)))
            frames.append(telemetry_proto(f, mps, lat, lon, heading, gear=0 if parked else 1))
        t_total += seconds
        for cam in cams:
            write_clip(
                root / folder / f"{ts}-{cam}.mp4",
                seconds,
                fps,
                size,
                CAM_HUES[cam],
                frames if cam == "front" else None,
            )
    if event is not None:
        (root / folder / "event.json").write_text(json.dumps(event))
    return stamps


FOUR = ["front", "back", "left_repeater", "right_repeater"]
SIX = ["front", "back", "left_repeater", "right_repeater", "left_pillar", "right_pillar"]


def make_teslacam(root: Path, seconds: float = 60, fps: int = 24, size: str = "480x360", small: bool = False) -> Path:
    """A TeslaCam tree with three drives. small=True: one short 4-camera drive (for tests)."""
    root = Path(root)
    if small:
        make_drive(root, "RecentClips", dt.datetime(2026, 1, 2, 3, 4, 5), 2, FOUR, seconds, fps, size)
        return root
    make_drive(root, "RecentClips", dt.datetime(2026, 11, 21, 17, 42, 10), 4, FOUR, seconds, fps, size, base_mph=27)
    ev_start = dt.datetime(2026, 11, 20, 8, 15, 3)
    make_drive(
        root,
        f"SavedClips/{ev_start.strftime('%Y-%m-%d_%H-%M-%S')}",
        ev_start,
        2,
        SIX,
        seconds,
        fps,
        size,
        base_mph=33,
        lat0=37.8021,
        lon0=-122.4488,
        heading0=110,
        event={"timestamp": ev_start.isoformat(), "city": "San Francisco", "reason": "user_interaction_dashcam"},
    )
    s_start = dt.datetime(2026, 11, 19, 22, 3, 41)
    make_drive(
        root,
        f"SentryClips/{s_start.strftime('%Y-%m-%d_%H-%M-%S')}",
        s_start,
        1,
        FOUR,
        seconds,
        fps,
        size,
        parked=True,
        event={"timestamp": s_start.isoformat(), "reason": "sentry_aware_object_detection"},
    )
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", help="folder to create (e.g. /tmp/TeslaCam)")
    ap.add_argument("--seconds", type=float, default=60, help="clip length")
    ap.add_argument("--small", action="store_true", help="one short drive (fast)")
    a = ap.parse_args()
    print(make_teslacam(Path(a.dest), a.seconds, small=a.small))

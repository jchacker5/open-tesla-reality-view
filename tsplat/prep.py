"""Turn one 6-camera clip window into a COLMAP-ready dataset + viewer assets.

work/<name>/
  images/<cam>/00000.jpg          frames sampled at --fps from every camera
  masks/<cam>/00000.jpg.png  COLMAP masks (black = ignore): ego-car body (+ movers if requested)
  rig_config.json            approximate Tesla camera layout, refined by COLMAP bundle adjustment
  pairs.txt                  rig-aware match list (same cam over time + overlapping neighbours)
  viewer/cams/<cam>.mp4      trimmed, downscaled clips for the side panel
  viewer/telemetry.json      SEI speed/GPS/steering at 10 Hz (if the clip has it)
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening, median_filter

from . import sei
from .ingest import CAMS

# Which cameras can see the same scene (spatially, allowing for car motion over time).
NEIGHBOURS = {
    "front": ["left_pillar", "right_pillar"],
    "left_pillar": ["front", "left_repeater"],
    "right_pillar": ["front", "right_repeater"],
    "left_repeater": ["left_pillar", "back"],
    "right_repeater": ["right_pillar", "back"],
    "back": ["left_repeater", "right_repeater"],
}
# 4-camera cars (no pillar cams, e.g. many HW3 cars): the front and repeater views only share scenery
# after the car drives past it, so they are paired across a wider time window.
NEIGHBOURS_4CAM = {
    "front": ["left_repeater", "right_repeater"],
    "left_repeater": ["front", "back"],
    "right_repeater": ["front", "back"],
    "back": ["left_repeater", "right_repeater"],
}


def neighbours(cams: list[str]) -> dict[str, list[str]]:
    table = NEIGHBOURS if {"left_pillar", "right_pillar"} & set(cams) else NEIGHBOURS_4CAM
    return {c: [n for n in table.get(c, []) if n in cams] for c in cams}


# Approximate camera layout (yaw degrees, right = +; position metres in the front camera's frame:
# x right, y down, z forward). Only a starting point: COLMAP refines it.
RIG = {
    "front": (0, (0.0, 0.0, 0.0)),
    "left_pillar": (-60, (-0.85, 0.15, -1.0)),
    "right_pillar": (60, (0.85, 0.15, -1.0)),
    "left_repeater": (-140, (-0.95, 0.55, 0.9)),
    "right_repeater": (140, (0.95, 0.55, 0.9)),
    "back": (180, (0.0, 0.55, -3.0)),
}
MOVER_CLASSES = {0, 1, 2, 3, 5, 7}  # COCO: person, bicycle, car, motorcycle, bus, truck


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def video_fps(path: str) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    num, _, den = r.stdout.strip().partition("/")
    return float(num) / float(den or 1)


def extract_frames(cams: dict, out: Path, start: float, dur: float, fps: float, width: int) -> None:
    for cam, path in cams.items():
        d = out / "images" / cam
        d.mkdir(parents=True, exist_ok=True)
        ff(
            "-ss",
            str(start),
            "-t",
            str(dur),
            "-i",
            path,
            "-vf",
            f"fps={fps},scale='min({width},iw)':-2",
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(d / "%05d.jpg"),
        )


def _body_region(cam: str, h: int, w: int) -> np.ndarray:
    """Where the car's own body can appear in each camera (keeps sky out of the mask)."""
    region = np.zeros((h, w), bool)
    if cam == "back":
        region[int(h * 0.60) :] = True
    elif cam in ("left_pillar", "right_pillar", "front"):
        region[int(h * 0.80) :] = True
    elif cam in ("left_repeater", "right_repeater"):  # body hugs one side edge; check both
        region[int(h * 0.15) :, : int(w * 0.25)] = True
        region[int(h * 0.15) :, int(w * 0.75) :] = True
    return region


def _to_keep_mask(body: np.ndarray, region: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    body = binary_closing(binary_opening(body & region, iterations=2), iterations=4) & region
    keep = np.where(body, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(keep).resize(size, Image.NEAREST))


def _gray(src, size=(320, 240)) -> np.ndarray:
    return np.asarray(Image.open(src).convert("L").resize(size), np.float32)


def build_ego_masks(index: str, out_dir: str, max_clips: int = 40) -> None:
    """Best ego mask: the body looks identical across *different drives*, everything else doesn't.
    Grabs one mid-clip frame per camera from up to `max_clips` clips and keeps low-variance pixels."""
    clips = [c for c in json.loads(Path(index).read_text()) if len(c["cams"]) >= 4]
    clips = clips[:: max(1, len(clips) // max_clips)][:max_clips]
    if len(clips) < 8:
        raise SystemExit(f"need >= 8 clips from different places for a good ego mask, have {len(clips)}")
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    for cam in CAMS:
        srcs = [c["cams"][cam] for c in clips if cam in c["cams"]]
        if len(srcs) < 8:
            continue
        stack, size = [], None
        for src in srcs:
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-ss",
                    "20",
                    "-i",
                    src,
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ],
                capture_output=True,
            )
            if not r.stdout:
                continue
            im = Image.open(io.BytesIO(r.stdout))
            size = size or im.size
            stack.append(_gray(im))
        if len(stack) < 8:
            continue
        spread = np.median(np.abs(np.stack(stack) - np.median(stack, 0)), 0)
        still = median_filter(spread, size=5) < 6.0
        Image.fromarray(_to_keep_mask(still, _body_region(cam, *still.shape), size)).save(od / f"{cam}.png")
        print(f"ego mask {cam}: {len(stack)} clips")


def ego_mask(frames: list[Path], cam: str, prebuilt: Path | None = None) -> np.ndarray:
    """255 = keep. Uses data/teslacam/egomask/<cam>.png (from `tsplat egomask`) when present;
    otherwise guesses from this window: pixels that barely change while driving."""
    size = Image.open(frames[0]).size
    if prebuilt and (prebuilt / f"{cam}.png").exists():
        return np.asarray(Image.open(prebuilt / f"{cam}.png").convert("L").resize(size, Image.NEAREST))
    stack = np.stack([_gray(f) for f in frames[:: max(1, len(frames) // 24)]])
    if len(stack) < 6:  # not enough motion to tell; don't mask
        return np.full(size[::-1], 255, np.uint8)
    # Median frame-to-frame change: robust to slow lighting drift and the odd passing car.
    diff = np.median(np.abs(np.diff(stack, axis=0)), axis=0)
    still = median_filter(diff, size=5) < 2.5
    return _to_keep_mask(still, _body_region(cam, *still.shape), size)


def mover_masks(images: list[Path]):
    """Yield (image, mask) where moving-capable objects are blacked out (needs ultralytics)."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("--mask-movers needs: uv sync --extra movers") from e
    model = YOLO("yolo11n-seg.pt")
    for img in images:
        res = model.predict(str(img), verbose=False, retina_masks=True)[0]
        h, w = res.orig_shape
        m = np.full((h, w), 255, np.uint8)
        if res.masks is not None:
            for cls, mk in zip(res.boxes.cls.tolist(), res.masks.data.cpu().numpy(), strict=False):
                if int(cls) in MOVER_CLASSES:
                    m[mk > 0.5] = 0
        yield img, m


def write_masks(out: Path, cams: list[str], movers: bool, prebuilt: Path | None) -> None:
    for cam in cams:
        frames = sorted((out / "images" / cam).glob("*.jpg"))
        base = ego_mask(frames, cam, prebuilt)
        md = out / "masks" / cam
        md.mkdir(parents=True, exist_ok=True)
        pairs = mover_masks(frames) if movers else ((f, None) for f in frames)
        for f, mv in pairs:
            m = base if mv is None else np.minimum(base, mv)
            Image.fromarray(m).save(md / f"{f.name}.png")


def write_pairs(out: Path, cams: list[str], same_k: int = 5, cross_k: int = 8) -> int:
    nbrs = neighbours(cams)
    if not ({"left_pillar", "right_pillar"} & set(cams)):
        cross_k *= 2
    n = {c: len(list((out / "images" / c).glob("*.jpg"))) for c in cams}
    pairs: set[tuple[str, str]] = set()

    def add(a: str, b: str) -> None:
        if a != b:
            pairs.add((a, b) if a < b else (b, a))

    for c in cams:
        for t in range(n[c]):
            me = f"{c}/{t:05d}.jpg"
            for dt in range(1, same_k + 1):
                if t + dt < n[c]:
                    add(me, f"{c}/{t + dt:05d}.jpg")
            for nb in nbrs[c]:
                for t2 in range(max(0, t - cross_k), min(n[nb], t + cross_k + 1)):
                    add(me, f"{nb}/{t2:05d}.jpg")
    (out / "pairs.txt").write_text("\n".join(f"{a} {b}" for a, b in sorted(pairs)) + "\n")
    return len(pairs)


def write_rig_config(out: Path, cams: list[str]) -> None:
    ref = "front" if "front" in cams else cams[0]
    entries = []
    for cam in cams:
        yaw, pos = RIG[cam]
        yaw0 = RIG[ref][0]
        th = np.radians(yaw - yaw0)
        e = {"image_prefix": f"{cam}/"}
        if cam == ref:
            e["ref_sensor"] = True
        else:
            # cam_from_rig = Ry(-th); translation = -R @ position
            R = np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0], [np.sin(th), 0, np.cos(th)]])
            e["cam_from_rig_rotation"] = [float(np.cos(-th / 2)), 0.0, float(np.sin(-th / 2)), 0.0]
            e["cam_from_rig_translation"] = (-R @ (np.array(pos) - np.array(RIG[ref][1]))).round(4).tolist()
        entries.append(e)
    (out / "rig_config.json").write_text(json.dumps([{"cameras": entries}], indent=1))


def write_viewer_assets(cams: dict, out: Path, start: float, dur: float) -> None:
    vd = out / "viewer" / "cams"
    vd.mkdir(parents=True, exist_ok=True)
    for cam, path in cams.items():
        ff(
            "-ss",
            str(start),
            "-t",
            str(dur),
            "-i",
            path,
            "-an",
            "-vf",
            "scale=640:-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-g",
            "15",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(vd / f"{cam}.mp4"),
        )

    rows = sei.extract(cams["front"]) if "front" in cams else []
    tel = []
    if rows:
        fps = video_fps(cams["front"])
        seq0 = rows[0]["frame_seq_no"]
        last = -1.0
        for r in rows:
            t = (r["frame_seq_no"] - seq0) / fps - start
            if 0 <= t <= dur and t - last >= 0.1:
                last = t
                tel.append(
                    {
                        "t": round(t, 2),
                        "mph": round(r.get("vehicle_speed_mps", 0) * 2.23694, 1),
                        "lat": r.get("latitude_deg"),
                        "lon": r.get("longitude_deg"),
                        "heading": r.get("heading_deg"),
                        "steer": r.get("steering_wheel_angle"),
                        "gear": r.get("gear"),
                        "autopilot": r.get("autopilot"),
                        "brake": r.get("brake_applied", False),
                        "blinkL": r.get("blinker_on_left", False),
                        "blinkR": r.get("blinker_on_right", False),
                    }
                )
    (out / "viewer" / "telemetry.json").write_text(json.dumps(tel))
    print(f"telemetry: {len(tel)} samples" if tel else "telemetry: none in this clip (needs fw 2025.44.25+)")


def run(
    index: str,
    clip_id: str,
    name: str,
    start: float,
    dur: float,
    fps: float,
    width: int,
    movers: bool,
    workdir: str = "work",
) -> Path:
    clips = json.loads(Path(index).read_text())
    clip = next((c for c in clips if c["id"] == clip_id or c["timestamp"] == clip_id), None)
    if not clip:
        raise SystemExit(f"clip {clip_id!r} not in {index}; run `tsplat list`")
    cams = {c: p for c, p in clip["cams"].items() if c in CAMS}
    out = Path(workdir) / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] {len(cams)} cams, {start}s +{dur}s @ {fps} fps")
    extract_frames(cams, out, start, dur, fps, width)
    egodir = Path(index).parent / "egomask"
    print("ego mask: " + ("prebuilt" if egodir.exists() else "guessed from window (run `tsplat egomask` for better)"))
    write_masks(out, list(cams), movers, egodir)
    print(f"pairs: {write_pairs(out, list(cams))}")
    write_rig_config(out, list(cams))
    write_viewer_assets(cams, out, start, dur)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "clip": clip["id"],
                "timestamp": clip["timestamp"],
                "start": start,
                "dur": dur,
                "fps": fps,
                "cams": list(cams),
            },
            indent=1,
        )
    )
    return out

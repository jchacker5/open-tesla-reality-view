"""What a studio build runs (`otrv prep` -> `otrv recon` -> `otrv bench` as subprocesses), and the
finished builds ("cities") it leaves in work/<name>/."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import footage
from .jobs import Job, Phase


def otrv(*args: str) -> list[str]:
    """The otrv CLI as a subprocess argv, using this interpreter (works without the console script)."""
    return [sys.executable, "-m", "tsplat", *args]


def frames_per_cam(spec: dict) -> int:
    return max(1, round(float(spec["dur"]) * float(spec["fps"])))


def units(job: Job) -> tuple[int, int, int | None]:
    """(images, training steps, mapper total). With the rig, COLMAP registers frames (one per timestamp)."""
    spec = job.spec
    n = frames_per_cam(spec)
    return n * len(spec["clip"]["cams"]), int(spec["steps"]), n


def probe(job: Job, stage: str) -> tuple[int, int] | None:
    """Progress for stages whose tools print none: count the files prep has written so far."""
    if stage not in ("frames", "masks"):
        return None
    total = units(job)[0]
    pattern = "images/*/*.jpg" if stage == "frames" else "masks/*/*.png"
    return min(total, sum(1 for _ in job.dir.glob(pattern))), total


def _staged(src: Path, clip: dict) -> bool:
    idx = src / "index.json"
    if not idx.exists():
        return False
    try:
        cams = json.loads(idx.read_text())[0]["cams"]
    except (ValueError, KeyError, IndexError):
        return False
    return all(Path(cams.get(c, "")).is_file() for c in clip["cams"])


def plan(job: Job, data: Path) -> list[Phase]:
    spec, name, src = job.spec, job.name, job.dir / "_src"
    clip = spec["clip"]

    def copy_footage(job: Job, log) -> None:
        if _staged(src, clip):
            log(f"footage already staged in {src}")
            return
        footage.stage_clip(clip, src, data, log, cancelled=lambda: job.cancel_requested)

    prep = otrv("--data", str(src), "prep", clip["id"], "--name", name)
    prep += ["--start", f"{spec['start']:g}", "--dur", f"{spec['dur']:g}", "--fps", f"{spec['fps']:g}"]
    prep += ["--width", str(spec.get("width", 1280))]
    if spec.get("mask_movers"):
        prep.append("--mask-movers")

    recon = otrv("recon", name, "--steps", str(spec["steps"]))
    if spec.get("eval"):
        recon.append("--eval")
    if spec.get("global_mapper"):
        recon.append("--global-mapper")
    sfm_done = (job.dir / "stages" / "undistort.done").exists() and (
        job.dir / "dense" / "sparse" / "images.txt"
    ).exists()
    recon_stages: tuple[str, ...] = ("features", "match", "map", "undistort", "train")
    if sfm_done:  # resume after SfM: retrain only
        recon, recon_stages = [*recon, "--skip-sfm"], ("train",)

    def finalize(job: Job, log) -> None:
        poster(job.dir, refresh=True)
        shutil.rmtree(src, ignore_errors=True)  # the copied clip; the viewer has its own trimmed videos
        (job.dir / "city.json").write_text(json.dumps({"built": time.time(), "spec": spec}, indent=1))
        log(f"city ready: {name}")

    return [
        Phase("prep", prep, ("frames", "masks"), before=copy_footage),
        Phase("recon", recon, recon_stages),
        Phase("bench", otrv("bench", name, "--note", "studio"), ("export",), required=False),
        Phase("finalize", finalize, ("export",)),
    ]


# ---- cities -----------------------------------------------------------------------------------------


def is_city(d: Path) -> bool:
    return (d / "viewer" / "scene.ply").is_file() and (d / "viewer" / "path.json").is_file()


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def dir_size(d: Path) -> int:
    total = 0
    for p in d.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def city(d: Path) -> dict:
    meta, metrics, built = _read(d / "meta.json"), _read(d / "metrics.json"), _read(d / "city.json")
    ply = d / "viewer" / "scene.ply"
    psnr = (metrics.get("psnr") or {}).get("overall")
    return {
        "name": d.name,
        "built": built.get("built") or ply.stat().st_mtime,
        "clip": meta.get("clip"),
        "timestamp": meta.get("timestamp"),
        "start": meta.get("start"),
        "dur": meta.get("dur"),
        "cams": meta.get("cams", []),
        "splat_bytes": ply.stat().st_size,
        "disk_bytes": dir_size(d),
        "psnr": psnr,
        "gates": metrics.get("gate_failures"),
        "steps": (_read(d / "recon_args.json")).get("steps"),
        "has_poster": poster(d) is not None,
    }


def cities(work: Path) -> list[dict]:
    if not work.is_dir():
        return []
    out = [city(d) for d in work.iterdir() if d.is_dir() and is_city(d)]
    return sorted(out, key=lambda c: c["built"], reverse=True)


def poster(d: Path, refresh: bool = False) -> Path | None:
    """First frame of the front camera clip the viewer plays: work/<name>/studio/poster.jpg (cached)."""
    out = d / "studio" / "poster.jpg"
    if out.exists() and not refresh:
        return out
    cams = d / "viewer" / "cams"
    src = next((cams / f"{c}.mp4" for c in ("front", "back") if (cams / f"{c}.mp4").exists()), None)
    if src is None:
        src = next(iter(sorted(cams.glob("*.mp4"))), None) if cams.is_dir() else None
    if src is None or not shutil.which("ffmpeg"):
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
        + ["-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", str(out)],
        capture_output=True,
        timeout=60,
    )
    return out if r.returncode == 0 and out.exists() else None

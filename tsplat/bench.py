"""Score a reconstruction and log it, so changes can be hill-climbed one lever at a time.

Metrics (all from files a `recon` run leaves behind):
  registration   share of sampled frames per camera that SfM placed in the model
  reproj_px      mean reprojection error of the sparse model
  psnr_db        masked PSNR of held-out views (every 8th frame), per camera and overall
                 (needs `recon --eval`)
  timings        seconds per stage, from timings.json
Every run is appended to work/ledger.jsonl, including failures, and gated (see GATES).
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

# A run must pass these before it can become the new best. Tune as the project improves.
GATES = {
    "min_registration": 0.90,  # every camera
    "max_reproj_px": 1.5,
}


def _image_names(dense: Path) -> list[str]:
    lines = (dense / "sparse" / "images.txt").read_text().splitlines()
    body = [line for line in lines if not line.startswith("#")]
    return [p.split()[9] for p in body[0::2] if len(p.split()) >= 10]


def registration(work: Path) -> dict[str, dict]:
    meta = json.loads((work / "meta.json").read_text())
    names = _image_names(work / "dense")
    out = {}
    for cam in meta["cams"]:
        total = len(list((work / "images" / cam).glob("*.jpg")))
        reg = sum(1 for n in names if n.startswith(f"{cam}/"))
        out[cam] = {"registered": reg, "total": total, "ratio": round(reg / total, 3) if total else 0.0}
    return out


def _sfm_model(work: Path) -> Path:
    """The model recon picked (largest registered set) in work/sparse/<n>; it keeps COLMAP's rig files."""
    models = [d for d in (work / "sparse").iterdir() if (d / "images.bin").exists()]
    return max(models, key=lambda d: (d / "images.bin").stat().st_size)


def reprojection(model: Path) -> dict:
    r = subprocess.run(["colmap", "model_analyzer", "--path", str(model)], capture_output=True, text=True)
    text = r.stdout + r.stderr

    def grab(key: str) -> float | None:
        m = re.search(rf"{key}:\s*([0-9.]+)", text)
        return float(m.group(1)) if m else None

    return {
        "reproj_px": grab("Mean reprojection error"),
        "points": grab("Points"),
        "obs_per_image": grab("Mean observations per image"),
    }


def psnr(work: Path) -> dict | None:
    """Masked PSNR of Brush's held-out renders (splat/eval_<step>/<cam>_<idx>.png) against the
    undistorted ground truth; masked-out pixels (ego car, movers) don't count."""
    evals = sorted((work / "splat").glob("eval_*"), key=lambda p: int(p.name.split("_")[1]))
    if not evals:
        return None
    dense = work / "dense" / "images"
    per_cam: dict[str, list[float]] = {}
    for render in evals[-1].glob("*.png"):
        cam = render.stem.rsplit("_", 1)[0]
        gt_path = dense / cam / f"{render.stem}.jpg"
        if not gt_path.exists():
            continue
        pred = Image.open(render).convert("RGB")
        gt = np.asarray(Image.open(gt_path).convert("RGB").resize(pred.size, Image.LANCZOS), np.float32) / 255
        pr = np.asarray(pred, np.float32) / 255
        mpath = dense / "masks" / f"{render.stem}.png"
        keep = (
            np.asarray(Image.open(mpath).convert("L").resize(pred.size, Image.NEAREST)) > 127
            if mpath.exists()
            else np.ones(gt.shape[:2], bool)
        )
        mse = float(((pr - gt) ** 2)[keep].mean()) if keep.any() else float("nan")
        per_cam.setdefault(cam, []).append(10 * np.log10(1 / max(mse, 1e-10)))
    if not per_cam:
        return None
    allv = [v for vs in per_cam.values() for v in vs]
    return {
        "step": int(evals[-1].name.split("_")[1]),
        "overall": round(float(np.mean(allv)), 2),
        "per_cam": {c: round(float(np.mean(v)), 2) for c, v in sorted(per_cam.items())},
        "views": len(allv),
    }


def _version(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).strip().splitlines()[0][:80]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "?"


def environment() -> dict:
    chip = (
        _version(["sysctl", "-n", "machdep.cpu.brand_string"])
        if platform.system() == "Darwin"
        else platform.processor()
    )
    try:
        ram = int(_version(["sysctl", "-n", "hw.memsize"])) // 2**30 if platform.system() == "Darwin" else None
    except ValueError:
        ram = None
    from .recon import BRUSH

    git = _version(["git", "rev-parse", "--short", "HEAD"])
    return {
        "chip": chip,
        "ram_gb": ram,
        "os": platform.platform(),
        "colmap": _version(["colmap", "-h"]),
        "brush": _version([BRUSH, "--version"]),
        "git": git,
    }


def gate(m: dict) -> list[str]:
    fails = []
    for cam, r in m["registration"].items():
        if r["ratio"] < GATES["min_registration"]:
            fails.append(f"{cam} registration {r['ratio']:.0%} < {GATES['min_registration']:.0%}")
    rp = m["sparse"].get("reproj_px")
    if rp is None or rp > GATES["max_reproj_px"]:
        fails.append(f"reprojection {rp} px > {GATES['max_reproj_px']}")
    return fails


def card(m: dict) -> str:
    reg = " · ".join(f"{c} {r['registered']}/{r['total']}" for c, r in m["registration"].items())
    p = m.get("psnr")
    t = m.get("timings", {})
    per_cam = ", ".join(f"{c} {v}" for c, v in p["per_cam"].items()) if p else ""
    lines = [
        f"### {m['name']}  ({m['env']['chip']}, {m['env']['ram_gb']} GB)",
        f"- registration: {reg}",
        f"- sparse: {m['sparse'].get('reproj_px')} px reprojection, {int(m['sparse'].get('points') or 0)} points",
        f"- held-out PSNR: {p['overall']} dB over {p['views']} views ({per_cam})"
        if p
        else "- held-out PSNR: n/a (run `recon --eval`)",
        "- time: " + ", ".join(f"{k} {v:.0f}s" for k, v in t.items()) if t else "- time: n/a",
        f"- config: {json.dumps(m['config'])}",
        f"- gates: {'PASS' if not m['gate_failures'] else 'FAIL: ' + '; '.join(m['gate_failures'])}",
    ]
    return "\n".join(lines)


def run(name: str, note: str = "", workdir: str = "work") -> dict:
    work = Path(workdir) / name
    meta = json.loads((work / "meta.json").read_text())
    args = json.loads((work / "recon_args.json").read_text()) if (work / "recon_args.json").exists() else {}
    m = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "name": name,
        "note": note,
        "config": {k: meta[k] for k in ("clip", "start", "dur", "fps") if k in meta}
        | {"cams": len(meta["cams"])}
        | args,
        "registration": registration(work),
        "sparse": reprojection(_sfm_model(work)),
        "psnr": psnr(work),
        "timings": json.loads((work / "timings.json").read_text()) if (work / "timings.json").exists() else {},
        "env": environment(),
    }
    m["gate_failures"] = gate(m)
    (work / "metrics.json").write_text(json.dumps(m, indent=1))
    with open(Path(workdir) / "ledger.jsonl", "a") as f:
        f.write(json.dumps(m) + "\n")
    print(card(m))
    return m

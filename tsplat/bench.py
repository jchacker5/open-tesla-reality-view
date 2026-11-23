"""Score a reconstruction and log it, so changes can be hill-climbed one lever at a time.

Metrics (all from files a `recon` run leaves behind):
  registration   share of sampled frames per camera that SfM placed in the model
  reproj_px      mean reprojection error of the sparse model
  psnr_db        masked PSNR of held-out views (every 8th frame), per camera and overall
                 (needs `recon --eval`)
  timings        seconds per stage, from timings.json
Every run is appended to work/ledger.jsonl, including failures, and gated (see GATES). A metric that
can't be computed (e.g. SfM failed, so there is no dense/) is recorded under "errors" and fails the gate.
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
    """Per camera: registered images that still exist in images/ (dense names are <cam>/<cam>_<orig>),
    out of the frames prep extracted. Raises if the model registers more images than exist, which means
    dense/ is from a different prep than images/."""
    meta = json.loads((work / "meta.json").read_text())
    names = set(_image_names(work / "dense"))
    out = {}
    for cam in meta["cams"]:
        frames = {p.name for p in (work / "images" / cam).glob("*.jpg")}
        prefix = f"{cam}/"
        in_cam = [n[len(prefix) :] for n in names if n.startswith(prefix)]
        reg = sum(1 for fn in in_cam if fn.removeprefix(f"{cam}_") in frames)
        total = len(frames)
        if reg > total or len(in_cam) > total:
            raise ValueError(f"{cam}: model registers {len(in_cam)} images but images/{cam} has {total}; stale dense/")
        out[cam] = {"registered": reg, "total": total, "ratio": round(reg / total, 3) if total else 0.0}
    return out


def _sfm_model(work: Path) -> Path:
    """The model recon picked (most registered images) in work/sparse/<n>; it keeps COLMAP's rig files."""
    from .recon import _registered

    sparse = work / "sparse"
    models = [d for d in sparse.iterdir() if _registered(d)] if sparse.is_dir() else []
    if not models:
        raise FileNotFoundError(f"no registered COLMAP model in {sparse}")
    return max(models, key=_registered)


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


def psnr(work: Path, recon_args: dict) -> dict | None:
    """Masked PSNR of Brush's held-out renders (splat/eval_<steps>/<cam>_<idx>.png) against the
    undistorted ground truth; masked-out pixels (ego car, movers) don't count. Only for runs trained
    with `recon --eval` (recon_args.json says so), and only from that run's final step."""
    if not recon_args.get("eval"):
        return None
    steps = recon_args["steps"]
    evals = work / "splat" / f"eval_{steps}"
    if not evals.is_dir():
        raise FileNotFoundError(f"recon ran with --eval but {evals} is missing (Brush saved no eval renders)")
    dense = work / "dense" / "images"
    per_cam: dict[str, list[float]] = {}
    for render in evals.glob("*.png"):
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
        raise FileNotFoundError(f"no eval render in {evals} matches an image in {dense}")
    allv = [v for vs in per_cam.values() for v in vs]
    return {
        "step": steps,
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
    from . import brush

    git = _version(["git", "rev-parse", "--short", "HEAD"])
    return {
        "chip": chip,
        "ram_gb": ram,
        "os": platform.platform(),
        "colmap": _version(["colmap", "-h"]),
        "brush": _version([brush.path(), "--version"]),
        "git": git,
    }


def gate(m: dict) -> list[str]:
    fails = [f"{metric}: {err}" for metric, err in m.get("errors", {}).items()]
    if not m.get("registration") and "registration" not in m.get("errors", {}):
        fails.append("registration: no cameras scored")
    for cam, r in (m.get("registration") or {}).items():
        if r["ratio"] > 1.0:
            fails.append(f"{cam} registration {r['registered']}/{r['total']} > 100% (stale model)")
        elif r["ratio"] < GATES["min_registration"]:
            fails.append(f"{cam} registration {r['ratio']:.0%} < {GATES['min_registration']:.0%}")
    rp = (m.get("sparse") or {}).get("reproj_px")
    if rp is None or rp > GATES["max_reproj_px"]:
        fails.append(f"reprojection {rp} px > {GATES['max_reproj_px']}")
    return fails


def card(m: dict) -> str:
    reg = " · ".join(f"{c} {r['registered']}/{r['total']}" for c, r in (m.get("registration") or {}).items())
    p = m.get("psnr")
    t = m.get("timings") or {}
    sp = m.get("sparse") or {}
    per_cam = ", ".join(f"{c} {v}" for c, v in p["per_cam"].items()) if p else ""
    env = m.get("env") or {}
    lines = [
        f"### {m['name']}  ({env.get('chip', '?')}, {env.get('ram_gb', '?')} GB)",
        f"- registration: {reg or 'n/a'}",
        f"- sparse: {sp.get('reproj_px')} px reprojection, {int(sp.get('points') or 0)} points",
        f"- held-out PSNR: {p['overall']} dB over {p['views']} views ({per_cam})"
        if p
        else "- held-out PSNR: n/a (run `otrv recon --eval`)",
        "- time: " + ", ".join(f"{k} {v:.0f}s" for k, v in t.items()) if t else "- time: n/a",
        f"- config: {json.dumps(m.get('config', {}))}",
        f"- gates: {'PASS' if not m['gate_failures'] else 'FAIL: ' + '; '.join(m['gate_failures'])}",
    ]
    return "\n".join(lines)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def run(name: str, note: str = "", workdir: str = "work") -> dict:
    """Score one run and append it to work/ledger.jsonl. Always appends, even when every metric fails."""
    work = Path(workdir) / name
    m: dict = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": name, "note": note}
    errors: dict[str, str] = {}

    def attempt(metric: str, fn, default=None):
        try:
            return fn()
        except Exception as e:  # a broken metric is recorded, never fatal
            errors[metric] = f"{type(e).__name__}: {e}"
            return default

    meta = attempt("meta", lambda: json.loads((work / "meta.json").read_text()), {})
    args = attempt("recon_args", lambda: _read_json(work / "recon_args.json"), {})
    m["config"] = {k: meta[k] for k in ("clip", "start", "dur", "fps") if k in meta} | {
        "cams": len(meta.get("cams", []))
    }
    m["config"] |= args
    m["registration"] = attempt("registration", lambda: registration(work), {})
    m["sparse"] = attempt("sparse", lambda: reprojection(_sfm_model(work)), {})
    m["psnr"] = attempt("psnr", lambda: psnr(work, args))
    m["timings"] = attempt("timings", lambda: _read_json(work / "timings.json"), {})
    m["env"] = attempt("env", environment, {})
    m["errors"] = errors
    m["gate_failures"] = gate(m)
    if work.is_dir():
        (work / "metrics.json").write_text(json.dumps(m, indent=1))
    Path(workdir).mkdir(parents=True, exist_ok=True)
    with open(Path(workdir) / "ledger.jsonl", "a") as f:
        f.write(json.dumps(m) + "\n")
    print(card(m))
    return m

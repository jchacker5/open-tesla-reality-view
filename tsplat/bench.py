"""Score a reconstruction and log it, so changes can be hill-climbed one lever at a time.

Metrics (all from files a `recon` run leaves behind):
  registration   share of sampled frames per camera that SfM placed in the model
  reproj_px      mean reprojection error of the sparse model
  psnr_db        masked PSNR of held-out views, per camera and overall (needs `recon --eval`); ego/mover
                 masks and sky (when recon saved sky masks) don't count. Ground truth is the untouched
                 undistorted image in dense_base/ (before --color-norm or --sky rewrote dense/), so a
                 lever can't change the images it is scored against
  psnr_cc        the same PSNR after a per-image affine colour fit (3x4, least squares on the scored
                 pixels) of the render to the ground truth: geometry/texture quality with exposure and
                 white-balance differences factored out
  lpips          masked LPIPS (AlexNet) of the same views (needs `uv sync --extra quality`, else skipped)
  road           PSNR and LPIPS on the bottom 35% of each held-out view, inside the mask
  floaters       splats with opacity > 0.1 inside the ego swept tube (1 m either side of the path,
                 0.15-1.6 m above the road), in viewer/scene.ply and in the trained splat before clean-up
  haze           share of the total opacity in translucent splats (opacity < 0.35), before/after
                 clean-up, plus the splats each clean-up rule removed (clean.json)
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
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

# A run must pass these before it can become the new best. Tune as the project improves.
GATES = {
    "min_registration": 0.90,  # every camera
    "max_reproj_px": 1.5,
}
# prep settings (meta.json) copied into each ledger line's config, next to recon_args.json
PREP_CONFIG = ("clip", "start", "dur", "fps", "width", "sky_model", "mask_movers")


def _image_names(dense: Path) -> list[str]:
    lines = (dense / "sparse" / "images.txt").read_text().splitlines()
    body = [line for line in lines if not line.startswith("#")]
    return [p.split()[9] for p in body[0::2] if len(p.split()) >= 10]


def registration(work: Path) -> dict[str, dict]:
    """Per camera: registered images that still exist in images/ (dense names are <cam>/<cam>_<orig>),
    out of the frames prep extracted. Raises if the model registers more images than exist, which means
    dense/ is from a different prep than images/."""
    meta = json.loads((work / "meta.json").read_text())
    names = set(_image_names(gt_root(work)))  # dense/ may hold recon --eval's placeholders
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


ROAD_BAND = 0.35  # bottom share of each held-out view scored on its own: the near road and sidewalks


class _Lpips:
    """LPIPS (AlexNet) on MPS/CUDA/CPU via the `lpips` package (`uv sync --extra quality`).
    Returns a per-pixel distance map so it can be averaged inside a mask."""

    def __init__(self):
        import lpips
        import torch

        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        with warnings.catch_warnings():  # torchvision deprecation noise from inside lpips
            warnings.simplefilter("ignore", UserWarning)
            self.net = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(self.dev).eval()

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        t = self.torch

        def tensor(a: np.ndarray):
            return t.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None].float().to(self.dev) * 2 - 1

        with t.no_grad():
            d = self.net(tensor(pred), tensor(gt))
        return d[0, 0].float().cpu().numpy()


def lpips_model():
    """An LPIPS callable, or None when the `quality` extra isn't installed (the metric is skipped)."""
    try:
        return _Lpips()
    except ImportError:
        return None


def _gt_image(dense: Path, cam: str, stem: str) -> Path | None:
    for ext in (".jpg", ".png"):  # .png = RGBA training image (transparent sky)
        p = dense / "images" / cam / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _keep(dense: Path, stem: str, size: tuple[int, int]) -> tuple[np.ndarray, bool]:
    """Scored pixels: inside the ego/mover mask and, when recon saved a sky mask (dense/sky/), not sky.
    Sky is excluded whatever the run's --sky setting, so runs with and without it stay comparable."""
    keep = np.ones(size[::-1], bool)
    mpath = dense / "images" / "masks" / f"{stem}.png"
    if mpath.exists():
        keep &= np.asarray(Image.open(mpath).convert("L").resize(size, Image.NEAREST)) > 127
    spath = dense / "sky" / f"{stem}.png"
    if spath.exists():
        keep &= np.asarray(Image.open(spath).convert("L").resize(size, Image.NEAREST)) <= 127
    return keep, spath.exists()


def _psnr(pred: np.ndarray, gt: np.ndarray, keep: np.ndarray) -> float | None:
    if not keep.any():
        return None
    mse = float(((pred - gt) ** 2)[keep].mean())
    return float(10 * np.log10(1 / max(mse, 1e-10)))


def _colour_fit(pred: np.ndarray, gt: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """pred mapped through the affine colour transform (rgb -> A rgb + b) that best fits gt on `keep`."""
    if keep.sum() < 16:
        return pred
    X = np.concatenate([pred[keep], np.ones((int(keep.sum()), 1), np.float32)], 1)
    M = np.linalg.lstsq(X, gt[keep], rcond=None)[0]  # (4, 3)
    out = np.concatenate([pred, np.ones(pred.shape[:2] + (1,), np.float32)], 2) @ M
    return np.clip(out, 0, 1).astype(np.float32)


def gt_root(work: Path) -> Path:
    """Where held-out ground truth comes from: dense_base/ (undistorted, never edited by a lever), or
    dense/ for runs from before dense_base/ existed."""
    base = work / "dense_base"
    return base if (base / "images").is_dir() else work / "dense"


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _round(v: float | None, nd: int) -> float | None:
    return None if v is None else round(v, nd)


def view_metrics(work: Path, recon_args: dict, lpips_fn=None) -> dict | None:
    """Masked PSNR and LPIPS of Brush's held-out renders (splat/eval_<steps>/<cam>_<idx>.png) against
    the undistorted ground truth, overall and on the near-road band (bottom ROAD_BAND of each view).
    Masked-out pixels (ego car, movers, sky) don't count. Only for runs trained with `recon --eval`
    (recon_args.json says so), and only from that run's final step. lpips_fn=None skips LPIPS."""
    if not recon_args.get("eval"):
        return None
    steps = recon_args["steps"]
    evals = work / "splat" / f"eval_{steps}"
    if not evals.is_dir():
        raise FileNotFoundError(f"recon ran with --eval but {evals} is missing (Brush saved no eval renders)")
    dense = gt_root(work)
    rows: dict[str, list[dict]] = {}
    sky_excluded = False
    for render in sorted(evals.glob("*.png")):
        cam = render.stem.rsplit("_", 1)[0]
        gt_path = _gt_image(dense, cam, render.stem)
        if gt_path is None:
            continue
        pred_img = Image.open(render).convert("RGB")
        gt = np.asarray(Image.open(gt_path).convert("RGB").resize(pred_img.size, Image.LANCZOS), np.float32) / 255
        pred = np.asarray(pred_img, np.float32) / 255
        keep, has_sky = _keep(dense, render.stem, pred_img.size)
        sky_excluded |= has_sky
        band = np.zeros_like(keep)
        band[int(round(keep.shape[0] * (1 - ROAD_BAND))) :] = True
        r = {
            "psnr": _psnr(pred, gt, keep),
            "psnr_cc": _psnr(_colour_fit(pred, gt, keep), gt, keep),
            "road_psnr": _psnr(pred, gt, keep & band),
        }
        if lpips_fn is not None:
            d = lpips_fn(pred, gt)
            r["lpips"] = float(d[keep].mean()) if keep.any() else None
            r["road_lpips"] = float(d[keep & band].mean()) if (keep & band).any() else None
        rows.setdefault(cam, []).append(r)
    if not rows:
        raise FileNotFoundError(f"no eval render in {evals} matches an image in {dense / 'images'}")
    split = json.loads((work / "eval.json").read_text()) if (work / "eval.json").exists() else None
    if split is not None:  # Brush must have held out exactly the views recon chose
        rendered = {r.stem for r in evals.glob("*.png")}
        extra, missing = rendered - set(split["eval"]), set(split["eval"]) - rendered
        if extra or missing:
            raise ValueError(
                f"Brush's eval views differ from eval.json: {len(extra)} unexpected "
                f"({', '.join(sorted(extra)[:3])}), {len(missing)} missing ({', '.join(sorted(missing)[:3])})"
            )

    def agg(key: str, nd: int) -> tuple[float | None, dict]:
        per = {c: _round(_mean([r.get(key) for r in rs]), nd) for c, rs in sorted(rows.items())}
        overall = _round(_mean([r.get(key) for rs in rows.values() for r in rs]), nd)
        return overall, {c: v for c, v in per.items() if v is not None}

    p_all, p_cam = agg("psnr", 2)
    views = sum(len(v) for v in rows.values())
    out = {
        "psnr": {
            "step": steps,
            "overall": p_all,
            "per_cam": p_cam,
            "views": views,
            "sky_excluded": sky_excluded,
            "gt": dense.name,
            "split": "time" if split else "every 8th image",
            "guard": len(split["guard"]) if split else 0,
        },
        "psnr_cc": dict(zip(("overall", "per_cam"), agg("psnr_cc", 2), strict=True)),
        "lpips": None,
        "road": {"band": ROAD_BAND, "psnr": agg("road_psnr", 2)[0], "lpips": None},
    }
    if lpips_fn is not None:
        l_all, l_cam = agg("lpips", 4)
        out["lpips"] = {"overall": l_all, "per_cam": l_cam, "net": "alex"}
        out["road"]["lpips"] = agg("road_lpips", 4)[0]
    return out


def psnr(work: Path, recon_args: dict) -> dict | None:
    """Masked held-out PSNR only (see view_metrics)."""
    v = view_metrics(work, recon_args)
    return v and v["psnr"]


def floater_stats(work: Path, recon_args: dict) -> dict | None:
    """Splats inside the ego swept tube (splats.floaters) in the scene the viewer shows and, when it
    exists, in the trained splat before pruning. None when recon didn't get as far as a scene."""
    from . import clean, splats

    scene_ply, path_json = work / "viewer" / "scene.ply", work / "viewer" / "path.json"
    if not (scene_ply.exists() and path_json.exists()):
        return None
    ego = splats.EgoPath.load(path_json)
    scene = splats.read_ply(scene_ply)
    out = {
        "scene": splats.floaters(scene, ego),
        "splats": len(scene),
        "translucent_mass": round(clean.translucent_mass(scene), 4),
    }
    raw = work / "splat" / f"splat_{recon_args.get('steps')}.ply"
    raw = raw if raw.exists() else work / "viewer" / "scene_raw.ply"
    if raw.exists():
        rp = splats.read_ply(raw)
        out |= {
            "raw": splats.floaters(rp, ego),
            "splats_raw": len(rp),
            "translucent_mass_raw": round(clean.translucent_mass(rp), 4),
        }
    return out


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
    lp = m.get("lpips")
    road = m.get("road") or {}
    fl = m.get("floaters") or {}
    t = m.get("timings") or {}
    sp = m.get("sparse") or {}
    per_cam = ", ".join(f"{c} {v}" for c, v in p["per_cam"].items()) if p else ""
    env = m.get("env") or {}
    if p:
        sky = ", sky excluded" if p.get("sky_excluded") else ""
        cc = m.get("psnr_cc") or {}
        psnr_line = (
            f"- held-out PSNR: {p['overall']} dB over {p['views']} views ({per_cam}){sky}; "
            f"colour-corrected {cc.get('overall')} dB; ground truth {p.get('gt', 'dense')}/"
        )
        lpips_line = (
            f"- held-out LPIPS: {lp['overall']} ({', '.join(f'{c} {v}' for c, v in lp['per_cam'].items())})"
            if lp
            else "- held-out LPIPS: n/a (needs `uv sync --extra quality`)"
        )
        road_line = (
            f"- near-road band (bottom {road.get('band', ROAD_BAND):.0%}): "
            f"PSNR {road.get('psnr')} dB, LPIPS {'n/a' if road.get('lpips') is None else road['lpips']}"
        )
    else:
        psnr_line, lpips_line, road_line = "- held-out PSNR: n/a (run `otrv recon --eval`)", None, None
    floater_line = (
        f"- floaters in ego tube: {fl['scene']} of {fl['splats']} splats"
        + (f" ({fl['raw']} of {fl['splats_raw']} before clean-up)" if "raw" in fl else "")
        if fl
        else "- floaters in ego tube: n/a"
    )
    haze_line = (
        f"- haze: {fl['translucent_mass']:.1%} of opacity mass in translucent splats (opacity < 0.35)"
        + (f", {fl['translucent_mass_raw']:.1%} before clean-up" if "translucent_mass_raw" in fl else "")
        if fl and "translucent_mass" in fl
        else None
    )
    cl = m.get("clean") or {}
    clean_line = (
        f"- clean: {cl['before']} -> {cl['after']} splats ("
        + (", ".join(f"{k} {v}" for k, v in cl.get("removed", {}).items()) or "no rules")
        + ")"
        if cl
        else None
    )
    lines = [
        f"### {m['name']}  ({env.get('chip', '?')}, {env.get('ram_gb', '?')} GB, {env.get('brush', 'brush ?')})",
        f"- registration: {reg or 'n/a'}",
        f"- sparse: {sp.get('reproj_px')} px reprojection, {int(sp.get('points') or 0)} points",
        psnr_line,
        lpips_line,
        road_line,
        floater_line,
        haze_line,
        clean_line,
        "- time: " + ", ".join(f"{k} {v:.0f}s" for k, v in t.items()) if t else "- time: n/a",
        f"- config: {json.dumps(m.get('config', {}))}",
        f"- gates: {'PASS' if not m['gate_failures'] else 'FAIL: ' + '; '.join(m['gate_failures'])}",
    ]
    return "\n".join(line for line in lines if line)


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
    m["config"] = {k: meta[k] for k in PREP_CONFIG if k in meta} | {"cams": len(meta.get("cams", []))}
    m["config"] |= args
    m["registration"] = attempt("registration", lambda: registration(work), {})
    m["sparse"] = attempt("sparse", lambda: reprojection(_sfm_model(work)), {})
    views = attempt("psnr", lambda: view_metrics(work, args, lpips_model() if args.get("eval") else None)) or {}
    m["psnr"], m["lpips"], m["road"] = views.get("psnr"), views.get("lpips"), views.get("road")
    m["psnr_cc"] = views.get("psnr_cc")
    m["floaters"] = attempt("floaters", lambda: floater_stats(work, args))
    cl = attempt("clean", lambda: _read_json(work / "clean.json"), {})
    m["clean"] = {k: cl[k] for k in ("before", "after", "removed") if k in cl}
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

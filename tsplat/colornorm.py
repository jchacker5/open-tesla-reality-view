"""Colour normalisation across cameras (`recon --color-norm`).

Tesla's cameras have their own exposure and white balance (the repeaters and the back camera
especially), so the same wall arrives in different colours from different cameras and Brush averages
them into a flat, blotchy middle. This matches every camera to the front camera in CIELAB:

    x' = (x - mu_cam(t)) * sd_front(t) / sd_cam(t) + mu_front(t)        per L, a, b channel

where mu/sd are the mean/std of the image's kept pixels (inside the ego/mover mask, not sky),
smoothed over a running window of WINDOW_S seconds. The smoothing is what makes this safe: per frame,
the back camera sees different things from the front one, but over a few seconds both cameras sweep
past the same street, so the remaining difference is mostly the camera. Gains and shifts are clamped.

Why not the render-based fit (a per-image 3x4 affine colour transform from a short pass-1 Brush
render to each photo): Brush 0.3's CLI only renders held-out eval views, not the training views, so
it would need our own splat rasteriser. This statistics match needs no extra training pass and runs
in seconds; the render fit stays a follow-up once tsplat has a renderer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

WINDOW_S = 4.0
MAX_GAIN = 1.4  # std ratio clamp, per channel
MAX_SHIFT = (15.0, 10.0, 10.0)  # L, a, b clamp on the mean shift

_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] (..., 3) -> CIELAB (D65)."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    lin = np.clip(xyz @ np.linalg.inv(_M).T, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def _keep(dense: Path, stem: str, size: tuple[int, int]) -> np.ndarray:
    keep = np.ones(size[::-1], bool)
    m = dense / "images" / "masks" / f"{stem}.png"
    if m.exists():
        keep &= np.asarray(Image.open(m).convert("L").resize(size, Image.NEAREST)) > 127
    s = dense / "sky" / f"{stem}.png"
    if s.exists():
        keep &= np.asarray(Image.open(s).convert("L").resize(size, Image.NEAREST)) <= 127
    return keep


def _smooth(idx: np.ndarray, vals: np.ndarray, at: np.ndarray, half: float) -> np.ndarray:
    """Mean of vals over frames within +-half of each `at` (falls back to the nearest frame)."""
    out = np.empty((len(at), vals.shape[1]))
    for k, t in enumerate(at):
        sel = np.abs(idx - t) <= half
        out[k] = vals[sel].mean(0) if sel.any() else vals[np.argmin(np.abs(idx - t))]
    return out


def run(dense: Path, fps: float, window_s: float = WINDOW_S) -> dict[str, dict]:
    """Rewrite dense/images/<cam>/*.jpg in place (call on a fresh dense/; recon restores it from
    dense_base/ every run). Returns per-camera mean gain/shift applied, for the log."""
    images = sorted((dense / "images").glob("*/*.jpg"))
    images = [p for p in images if p.parent.name != "masks"]
    by_cam: dict[str, list[tuple[int, Path]]] = {}
    for p in images:
        cam, _, idx = p.stem.rpartition("_")
        by_cam.setdefault(cam, []).append((int(idx), p))
    if not by_cam:
        return {}
    ref = "front" if "front" in by_cam else sorted(by_cam)[0]
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cam, items in by_cam.items():
        rows = []
        for idx, p in sorted(items):
            img = Image.open(p).convert("RGB")
            lab = rgb_to_lab(np.asarray(img.resize((img.width // 4 or 1, img.height // 4 or 1)), np.float64) / 255)
            k = _keep(dense, p.stem, (lab.shape[1], lab.shape[0]))
            px = lab[k] if k.mean() > 0.05 else lab.reshape(-1, 3)
            rows.append([idx, *px.mean(0), *px.std(0)])
        stats[cam] = np.array(rows)[:, 0], np.array(rows)[:, 1:]
    half = window_s * fps / 2
    ref_idx, ref_vals = stats[ref]
    report = {}
    for cam, items in by_cam.items():
        idx, vals = stats[cam]
        src = _smooth(idx, vals, idx, half)
        tgt = _smooth(ref_idx, ref_vals, idx, half)
        gain = np.clip(tgt[:, 3:] / np.maximum(src[:, 3:], 1e-6), 1 / MAX_GAIN, MAX_GAIN)
        shift = np.clip(tgt[:, :3] - src[:, :3], -np.array(MAX_SHIFT), MAX_SHIFT)
        if cam != ref:
            order = {i: k for k, i in enumerate(idx.astype(int))}
            for i, p in items:
                k = order[i]
                rgb = np.asarray(Image.open(p).convert("RGB"), np.float64) / 255
                lab = rgb_to_lab(rgb)
                mu = src[k, :3]
                lab = (lab - mu) * gain[k] + mu + shift[k]
                out = (np.clip(lab_to_rgb(lab), 0, 1) * 255 + 0.5).astype(np.uint8)
                Image.fromarray(out).save(p, quality=95)
        report[cam] = {"gain": gain.mean(0).round(3).tolist(), "shift": shift.mean(0).round(2).tolist()}
    print("color norm (LAB, to " + ref + "): " + ", ".join(f"{c} dL {r['shift'][0]:+.1f}" for c, r in report.items()))
    return report

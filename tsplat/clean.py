"""Post-train clean-up of the splat: ego-path pruning and dehazing (`recon` runs it; `otrv clean`
re-runs it with other settings without retraining).

Haze is what makes a splat look milky: many translucent splats layered over the road and the sky's
edge, and long translucent "needles". Every rule below is a flag, and a splat is attributed to the
first rule that removes it, in this order:

  prune (recon --prune, default on)
    tube           inside the ego swept tube (1 m either side of the path, 0.15-1.6 m above the road)
    giant          any axis > 3 m and opacity < 0.2
  dehaze (--dehaze, default on)
    low_opacity    opacity < --min-opacity (0.04): contributes almost nothing on its own, but stacks
    translucent    largest axis > --translucent-scale (0.8 m) and opacity < --translucent-opacity (0.35)
    needle         largest/smallest axis > --needle-ratio (25) and opacity < --needle-opacity (0.5)
    ground_fog     more than --ground-fog-m (40 m) from the path and within 1 m of road height
    sky_fog        only with --dehaze-sky, and only when the sky wasn't trained transparent: above
                   eye level, nearer than --sky-fog-m (60 m), pale (saturation < 0.15) and bright
                   (> 0.6): sky painted at a finite depth

Writes viewer/scene.ply (cleaned), viewer/scene_raw.ply (the trained splat, untouched) and
clean.json (counts per rule), and preserves every PLY property (SH included).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import splats

TRANSLUCENT = 0.35  # bench: opacity below this counts as translucent haze mass


@dataclass
class Rules:
    prune: bool = True
    dehaze: bool = True
    min_opacity: float = 0.04
    translucent_scale_m: float = 0.8
    translucent_opacity: float = 0.35
    needle_ratio: float = 25.0
    needle_opacity: float = 0.5
    ground_fog_m: float = 40.0
    ground_fog_band_m: float = 1.0
    sky_fog: bool = False
    sky_fog_m: float = 60.0
    sky_fog_saturation: float = 0.15
    sky_fog_brightness: float = 0.6


def masks(ply: splats.Ply, ego: splats.EgoPath, rules: Rules) -> dict[str, np.ndarray]:
    """Boolean 'remove' mask per rule (overlapping; see clean() for attribution)."""
    n = len(ply)
    out: dict[str, np.ndarray] = {}
    op = ply.opacity()
    big = ply.max_scale() * ego.mpu
    lateral, height, _ = ego.nearest_road(ply.xyz()) if n else (np.zeros(0), np.zeros(0), None)
    if rules.prune:
        out["tube"] = (lateral <= splats.TUBE_RADIUS_M) & (height >= splats.TUBE_LO_M) & (height <= splats.TUBE_HI_M)
        out["giant"] = (big > 3.0) & (op < 0.2)
    if rules.dehaze:
        out["low_opacity"] = op < rules.min_opacity
        out["translucent"] = (big > rules.translucent_scale_m) & (op < rules.translucent_opacity)
        ratio = ply.max_scale() / np.maximum(ply.min_scale(), 1e-12)
        out["needle"] = (ratio > rules.needle_ratio) & (op < rules.needle_opacity)
        out["ground_fog"] = (lateral > rules.ground_fog_m) & (np.abs(height) < rules.ground_fog_band_m)
        if rules.sky_fog:
            rgb = ply.rgb()
            bright = rgb.max(1)
            sat = (bright - rgb.min(1)) / np.maximum(bright, 1e-6)
            out["sky_fog"] = (
                (height > splats.CAM_HEIGHT_M)
                & (lateral < rules.sky_fog_m)
                & (sat < rules.sky_fog_saturation)
                & (bright > rules.sky_fog_brightness)
            )
    return {k: v if len(v) == n else np.zeros(n, bool) for k, v in out.items()}


def clean(ply: splats.Ply, ego: splats.EgoPath, rules: Rules) -> tuple[splats.Ply, dict]:
    drop = np.zeros(len(ply), bool)
    removed = {}
    for name, m in masks(ply, ego, rules).items():
        removed[name] = int((m & ~drop).sum())
        drop |= m
    stats = {"before": len(ply), "after": int((~drop).sum()), "removed": removed}
    return ply.subset(~drop), stats


def translucent_mass(ply: splats.Ply, below: float = TRANSLUCENT) -> float:
    """Share of the total opacity that sits in translucent splats (opacity < below)."""
    op = ply.opacity()
    total = op.sum()
    return float(op[op < below].sum() / total) if total > 0 else 0.0


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.unlink(missing_ok=True)
    try:
        os.link(src, dest)  # no second copy of a large splat on disk
    except OSError:
        shutil.copy2(src, dest)


def run(work: Path, src: Path, rules: Rules) -> dict:
    """Clean `src` (the trained splat) into viewer/scene.ply; keep it as viewer/scene_raw.ply."""
    vd = work / "viewer"
    vd.mkdir(parents=True, exist_ok=True)
    raw = vd / "scene_raw.ply"
    if src.resolve() != raw.resolve():
        _link_or_copy(src, raw)
    ply = splats.read_ply(raw)
    kept, stats = clean(ply, splats.EgoPath.load(vd / "path.json"), rules)
    stats["rules"] = asdict(rules)
    stats["translucent_mass"] = round(translucent_mass(kept), 4)
    stats["translucent_mass_raw"] = round(translucent_mass(ply), 4)
    splats.write_ply(vd / "scene.ply", kept)
    (work / "clean.json").write_text(json.dumps(stats, indent=1))
    parts = ", ".join(f"{k} {v}" for k, v in stats["removed"].items())
    print(f"clean: {stats['before']} -> {stats['after']} splats ({parts or 'no rules'})")
    print(f"haze: translucent opacity mass {stats['translucent_mass_raw']:.1%} -> {stats['translucent_mass']:.1%}")
    return stats

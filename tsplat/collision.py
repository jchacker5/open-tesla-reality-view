"""Collision bake for the viewer's drive mode: viewer/collision.json (`recon` writes it last, after
clean-up; `otrv collision <name>` and `otrv clean` re-bake it from viewer/scene.ply).

Frame. Everything is in scene units (COLMAP's world units; `metresPerUnit` converts) in a horizontal
frame built from viewer/path.json:
  origin  the road under the first path pose (front camera minus path.json camHeight, fitted per
          scene, default 1.4 m, along `up`)
  up      path.json's `up`
  fwd     the path's overall direction (last pose minus first), made perpendicular to `up`
  right   fwd x up (to the driver's right)
A scene point p has grid coordinates x = (p - origin).right, z = (p - origin).fwd and height
y = (p - origin).up; back again, p = origin + x*right + y*up + z*fwd.

Grids. Cell (ix, iz) of a grid covers x in [x0 + ix*cell, x0 + (ix+1)*cell) and z likewise from z0;
arrays are row-major, value = array[iz * nx + ix]. Both grids cover the corridor: everything within
12 m (horizontally) of the path, plus its bounding box.

  height[]    0.5 m cells: ground height y. Per cell, the 10th percentile of the heights of the splats
              in it (opacity > 0.2, within 2 m of the path's road height, so trees and bridges don't
              count); cells with fewer than 4 such splats take the road height of the nearest path pose.
  wallDist[]  0.25 m cells: horizontal distance to the nearest wall, in scene units (0 = inside a wall).
              A wall cell holds at least 3 splats with opacity > 0.3 standing 0.5-3 m above the ground
              grid, after a one-cell morphological closing. The car's own path (1 m either side) is
              always free, and everything outside the corridor counts as wall, so drive mode stays on
              the reconstructed part of the scene.

Format: {cell, x0, z0, nx, nz, height[], wallCell, wx0, wz0, wnx, wnz, wallDist[],
         basis: {origin, right, up, fwd}, metresPerUnit, corridor, version}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt
from scipy.spatial import cKDTree

from . import splats

GROUND_CELL_M = 0.5
WALL_CELL_M = 0.25
CORRIDOR_M = 12.0
GROUND_PERCENTILE = 10
GROUND_MIN_SPLATS = 4
GROUND_MIN_OPACITY = 0.2
GROUND_BAND_M = 2.0  # ground candidates lie within this of the path's road height
WALL_LO_M, WALL_HI_M = 0.5, 3.0  # above the ground grid
WALL_MIN_OPACITY = 0.3
WALL_MIN_SPLATS = 3  # per 0.25 m cell
PATH_CLEAR_M = 1.0  # half the car's width: the path it actually drove is never a wall


@dataclass
class Grid:
    x0: float
    z0: float
    cell: float
    nx: int
    nz: int

    @classmethod
    def covering(cls, lo: np.ndarray, hi: np.ndarray, cell: float) -> Grid:
        nx, nz = (np.maximum(1, np.ceil((hi - lo) / cell)).astype(int)).tolist()
        return cls(float(lo[0]), float(lo[1]), cell, nx, nz)

    def index(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clipped (ix, iz) of each point."""
        ix = np.clip(np.floor((x - self.x0) / self.cell).astype(int), 0, self.nx - 1)
        iz = np.clip(np.floor((z - self.z0) / self.cell).astype(int), 0, self.nz - 1)
        return ix, iz

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        """(nz, nx) arrays of cell-centre x and z."""
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell
        zs = self.z0 + (np.arange(self.nz) + 0.5) * self.cell
        return np.meshgrid(xs, zs)


def frame(ego: splats.EgoPath) -> dict[str, np.ndarray]:
    right, fwd = ego.horizontal_basis()
    return {"origin": ego.road[0], "right": right, "up": ego.up, "fwd": fwd}


def to_frame(xyz: np.ndarray, basis: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, z) = coordinates along right, up, fwd from the origin, in scene units."""
    rel = np.asarray(xyz, float).reshape(-1, 3) - basis["origin"]
    return rel @ basis["right"], rel @ basis["up"], rel @ basis["fwd"]


def _percentile_per_cell(cell_ids: np.ndarray, values: np.ndarray, n_cells: int, pct: float, min_count: int):
    """Nearest-rank percentile of `values` per cell, NaN where a cell has fewer than min_count values."""
    out = np.full(n_cells, np.nan)
    if len(values) == 0:
        return out
    order = np.lexsort((values, cell_ids))
    ids, vals = cell_ids[order], values[order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    counts = np.diff(np.r_[starts, len(ids)])
    ok = counts >= min_count
    pick = starts + np.floor(pct / 100 * (counts - 1)).astype(int)
    out[ids[starts[ok]]] = vals[pick[ok]]
    return out


def _round(a: np.ndarray, sig: int = 5) -> list[float]:
    return [float(f"{v:.{sig}g}") for v in np.asarray(a, float).ravel()]


def bake(xyz: np.ndarray, opacity: np.ndarray, ego: splats.EgoPath, corridor_m: float = CORRIDOR_M) -> dict:
    """The collision document for splat centres `xyz` (scene units) with opacities in [0, 1]."""
    u = 1.0 / ego.mpu  # scene units per metre
    basis = frame(ego)
    road = ego.dense_road(step_m=0.25)
    rx, ry, rz = to_frame(road, basis)
    road_tree = cKDTree(np.stack([rx, rz], 1))
    corridor = corridor_m * u
    lo = np.array([rx.min(), rz.min()]) - corridor
    hi = np.array([rx.max(), rz.max()]) + corridor

    sx, sy, sz = to_frame(xyz, basis)
    op = np.asarray(opacity, float)
    lateral, near = road_tree.query(np.stack([sx, sz], 1), workers=-1) if len(sx) else (np.zeros(0), np.zeros(0, int))
    inside = lateral <= corridor

    # ground: low percentile of plausible ground splats per 0.5 m cell, else the path's road height
    g = Grid.covering(lo, hi, GROUND_CELL_M * u)
    cand = inside & (op > GROUND_MIN_OPACITY) & (np.abs(sy - ry[near]) < GROUND_BAND_M * u)
    gix, giz = g.index(sx[cand], sz[cand])
    height = _percentile_per_cell(giz * g.nx + gix, sy[cand], g.nx * g.nz, GROUND_PERCENTILE, GROUND_MIN_SPLATS)
    gcx, gcz = g.centres()
    _, gnear = road_tree.query(np.stack([gcx.ravel(), gcz.ravel()], 1), workers=-1)
    sparse = np.isnan(height)
    height[sparse] = ry[gnear[sparse]]

    # walls: dense, fairly opaque splats standing 0.5-3 m above the ground grid, per 0.25 m cell
    w = Grid.covering(lo, hi, WALL_CELL_M * u)
    gx, gz = g.index(sx, sz)
    above = sy - height[gz * g.nx + gx]
    solid = inside & (op > WALL_MIN_OPACITY) & (above >= WALL_LO_M * u) & (above <= WALL_HI_M * u)
    wix, wiz = w.index(sx[solid], sz[solid])
    counts = np.bincount(wiz * w.nx + wix, minlength=w.nx * w.nz).reshape(w.nz, w.nx)
    wall = np.pad(counts >= WALL_MIN_SPLATS, 1)
    wall = binary_closing(wall, structure=np.ones((3, 3), bool))[1:-1, 1:-1]
    wcx, wcz = w.centres()
    wlat, _ = road_tree.query(np.stack([wcx.ravel(), wcz.ravel()], 1), workers=-1)
    wlat = wlat.reshape(w.nz, w.nx)
    wall[wlat <= PATH_CLEAR_M * u] = False
    wall[wlat > corridor] = True
    dist = distance_transform_edt(~wall) * w.cell if wall.any() else np.full(wall.shape, corridor)

    return {
        "version": 1,
        "cell": g.cell,
        "x0": g.x0,
        "z0": g.z0,
        "nx": g.nx,
        "nz": g.nz,
        "height": _round(height),
        "wallCell": w.cell,
        "wx0": w.x0,
        "wz0": w.z0,
        "wnx": w.nx,
        "wnz": w.nz,
        "wallDist": _round(dist, 4),
        "basis": {k: np.round(v, 6).tolist() for k, v in basis.items()},
        "metresPerUnit": ego.mpu,
        "corridor": corridor,
        "stats": {
            "splats": int(len(sx)),
            "ground_from_splats": round(float(1 - sparse.mean()), 3),
            "wall_cells": int((wall & (wlat <= corridor)).sum()),
        },
    }


def run(work: Path, scene: Path | None = None) -> Path:
    """Bake viewer/collision.json from the cleaned splat (viewer/scene.ply) and viewer/path.json."""
    vd = work / "viewer"
    ply = splats.read_ply(scene or vd / "scene.ply")
    doc = bake(ply.xyz(), ply.opacity(), splats.EgoPath.load(vd / "path.json"))
    out = vd / "collision.json"
    out.write_text(json.dumps(doc, separators=(",", ":")))
    s = doc["stats"]
    print(
        f"collision: ground {doc['nx']}x{doc['nz']} cells of {GROUND_CELL_M} m "
        f"({s['ground_from_splats']:.0%} from splats, the rest from the path), "
        f"{s['wall_cells']} wall cells of {WALL_CELL_M} m -> {out}"
    )
    return out

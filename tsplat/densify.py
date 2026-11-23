"""Dense depth initialisation for Brush (`recon --dense-init`, needs `uv sync --extra quality`).

SfM leaves a few thousand points, almost none on the road (textureless asphalt), so Brush has to grow
the road from nothing and smears it. This adds two kinds of points to dense/sparse/points3D.txt:

1. Monocular depth: Depth Anything V2 Small (Apache-2.0) on every undistorted training image (MPS).
   Its output is inverse depth up to an unknown scale and shift, so each image gets its own
   1/z = s*d + b fitted to the COLMAP points it observes (RANSAC, then Huber IRLS). Masked (ego,
   movers) and sky pixels and depth edges are dropped; the rest are back-projected, then voxel-
   downsampled: 5 cm within 3 m of the path (the road the chase cam looks at), 15 cm elsewhere,
   averaging colour. Capped at MAX_POINTS.
2. A road ribbon: a 10 cm grid on the road plane along the path, +-6 m to each side, each point
   coloured from the nearest front-camera image that sees it unmasked.

The points are appended with fresh ids and an empty track. Brush 0.3's colmap reader accepts that
(the track is `parts[8..]` read in pairs, so zero pairs is fine) and initialises one splat per point
from xyz + rgb; it must not see blank lines, which it rejects (and then silently falls back to a
random init).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .recon import qvec_to_R
from .splats import EgoPath

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"  # Apache-2.0
MAX_POINTS = 1_500_000
NEAR_ROAD_M = 3.0
VOXEL_NEAR_M = 0.05
VOXEL_FAR_M = 0.15
MAX_DEPTH_M = 80.0
RIBBON_HALF_M = 6.0
RIBBON_STEP_M = 0.10
PIXELS_PER_IMAGE = 40_000  # back-projected per image before voxelisation (stride picked to match)


@dataclass
class View:
    name: str
    cam: int
    R: np.ndarray  # world -> camera
    t: np.ndarray
    xy: np.ndarray  # (k, 2) observed keypoints with a 3D point
    pid: np.ndarray  # (k,) their point ids

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t


def read_model(sparse: Path) -> tuple[dict, list[View], dict[int, np.ndarray]]:
    """cameras {id: (w, h, fx, fy, cx, cy)} (PINHOLE), views, points {id: xyz} from a TXT model."""
    cams = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        p = line.split()
        if p and not line.startswith("#"):
            cams[int(p[0])] = (int(p[2]), int(p[3]), *(float(v) for v in p[4:8]))
    rows = [line for line in (sparse / "images.txt").read_text().splitlines() if not line.startswith("#")]
    views = []
    for pose, obs in zip(rows[0::2], rows[1::2], strict=False):
        p = pose.split()
        o = np.array(obs.split(), float).reshape(-1, 3) if obs.strip() else np.zeros((0, 3))
        o = o[o[:, 2] >= 0]
        R = qvec_to_R([float(v) for v in p[1:5]])
        views.append(View(p[9], int(p[8]), R, np.array([float(v) for v in p[5:8]]), o[:, :2], o[:, 2].astype(int)))
    pts = {}
    for line in (sparse / "points3D.txt").read_text().splitlines():
        p = line.split()
        if p and not line.startswith("#"):
            pts[int(p[0])] = np.array([float(v) for v in p[1:4]])
    return cams, views, pts


def fit_inverse_depth(d: np.ndarray, inv_z: np.ndarray, iters: int = 200, seed: int = 0) -> tuple[float, float] | None:
    """Robust (s, b) with inv_z ~= s*d + b: RANSAC on pairs (inlier = within 10% of inv_z), then
    Huber-weighted least squares on the inliers. None if fewer than 8 points or no positive scale."""
    if len(d) < 8:
        return None
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(d), 2, replace=False)
        if abs(d[i] - d[j]) < 1e-9:
            continue
        s = (inv_z[i] - inv_z[j]) / (d[i] - d[j])
        b = inv_z[i] - s * d[i]
        n = int((np.abs(s * d + b - inv_z) < 0.1 * inv_z).sum())
        if s > 0 and n > best_n:
            best, best_n = (s, b), n
    if best is None or best_n < 6:
        return None
    s, b = best
    inl = np.abs(s * d + b - inv_z) < 0.1 * inv_z
    A = np.stack([d[inl], np.ones(inl.sum())], 1)
    y = inv_z[inl]
    w = np.ones(len(y))
    for _ in range(10):  # IRLS with a Huber loss (delta = 5% of inverse depth)
        sw = np.sqrt(w)
        s, b = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = np.abs(A @ [s, b] - y)
        delta = 0.05 * np.median(y)
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
    return (float(s), float(b)) if s > 0 else None


class DepthModel:
    """Depth Anything V2 Small via transformers: relative inverse depth at the image's resolution."""

    def __init__(self, model_id: str = MODEL_ID):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise SystemExit("--dense-init needs: uv sync --extra quality") from e
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.net = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.dev).eval()

    def __call__(self, img: Image.Image) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            x = self.proc(images=img.convert("RGB"), return_tensors="pt")["pixel_values"].to(self.dev)
            d = self.net(pixel_values=x).predicted_depth[:, None]
            d = t.nn.functional.interpolate(d, size=img.size[::-1], mode="bilinear", align_corners=False)
        return d[0, 0].float().cpu().numpy()


def _mask(path: Path, size: tuple[int, int], default: bool) -> np.ndarray:
    if not path.exists():
        return np.full(size[::-1], default)
    return np.asarray(Image.open(path).convert("L").resize(size, Image.NEAREST)) > 127


def _image_path(dense: Path, name: str) -> Path:
    return dense / "images" / name


def backproject(dense: Path, cams: dict, views: list[View], pts: dict, depth_fn, max_depth: float, report=print):
    """Per image: fit the depth, back-project kept pixels. Returns (xyz float64 (n,3), rgb uint8 (n,3))."""
    xyz_all, rgb_all, used = [], [], 0
    for v in views:
        w, h, fx, fy, cx, cy = cams[v.cam]
        img = Image.open(_image_path(dense, v.name)).convert("RGB")
        if img.size != (w, h):
            continue
        stem = Path(v.name).stem
        keep = _mask(dense / "images" / "masks" / f"{stem}.png", img.size, True)
        keep &= ~_mask(dense / "sky" / f"{stem}.png", img.size, False)
        known = np.array([pid in pts for pid in v.pid], bool)
        if known.sum() < 8:
            continue
        # Project the observed points with the current intrinsics rather than trusting POINTS2D, which
        # recon's small-image upscale (_ensure_min_size) leaves at the old resolution.
        Xc = np.stack([pts[pid] for pid in v.pid[known]]) @ v.R.T + v.t
        z = Xc[:, 2]
        zs = np.maximum(z, 1e-9)
        xy = np.stack([fx * Xc[:, 0] / zs + cx, fy * Xc[:, 1] / zs + cy], 1)
        ok = (z > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        d = depth_fn(img)
        dk = d[xy[ok, 1].astype(int), xy[ok, 0].astype(int)]
        fit = fit_inverse_depth(dk, 1.0 / z[ok])
        if fit is None:
            continue
        inv = fit[0] * d + fit[1]
        depth = np.where(inv > 1.0 / max_depth, 1.0 / np.maximum(inv, 1e-9), 0.0)
        logd = np.log(np.maximum(depth, 1e-9))
        edges = np.hypot(*np.gradient(logd)) > 0.05  # depth discontinuities: "flying pixels"
        good = keep & (depth > 0) & ~ndi.binary_dilation(edges)
        stride = max(1, int(np.sqrt(w * h / PIXELS_PER_IMAGE)))
        sub = np.zeros_like(good)
        sub[::stride, ::stride] = True
        ys, xs = np.nonzero(good & sub)
        zz = depth[ys, xs]
        cam_pts = np.stack([(xs + 0.5 - cx) / fx * zz, (ys + 0.5 - cy) / fy * zz, zz], 1)
        xyz_all.append((cam_pts - v.t) @ v.R)  # R^T (x - t)
        rgb_all.append(np.asarray(img)[ys, xs])
        used += 1
    report(f"dense init: depth fitted on {used}/{len(views)} images")
    if not xyz_all:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    return np.concatenate(xyz_all), np.concatenate(rgb_all)


def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean position and colour per voxel; `voxel` is the size per point (so near and far differ)."""
    if len(xyz) == 0:
        return xyz, rgb
    out_x, out_c = [], []
    for size in np.unique(voxel):
        sel = voxel == size
        keys = np.floor(xyz[sel] / size).astype(np.int64)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        sx = np.zeros((len(counts), 3))
        sc = np.zeros((len(counts), 3))
        np.add.at(sx, inv, xyz[sel])
        np.add.at(sc, inv, rgb[sel].astype(np.float64))
        out_x.append(sx / counts[:, None])
        out_c.append(np.round(sc / counts[:, None]).astype(np.uint8))
    return np.concatenate(out_x), np.concatenate(out_c)


def road_ribbon(dense: Path, cams: dict, views: list[View], ego: EgoPath, ref: str = "front"):
    """Road-plane grid along the path (+-RIBBON_HALF_M, RIBBON_STEP_M spacing), coloured from the
    closest `ref`-camera image that sees each point inside its frame and mask. Unseen points are dropped."""
    road = ego.dense_road(RIBBON_STEP_M)
    if len(road) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    fwd = np.gradient(road, axis=0)
    fwd -= np.outer(fwd @ ego.up, ego.up)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-12
    right = np.cross(fwd, ego.up)
    offs = np.arange(-RIBBON_HALF_M, RIBBON_HALF_M + 1e-9, RIBBON_STEP_M) / ego.mpu
    grid = (road[:, None, :] + offs[None, :, None] * right[:, None, :]).reshape(-1, 3)
    best = np.full(len(grid), np.inf)
    rgb = np.zeros((len(grid), 3), np.uint8)
    for v in views:
        if not v.name.startswith(ref + "/"):
            continue
        w, h, fx, fy, cx, cy = cams[v.cam]
        pc = grid @ v.R.T + v.t
        front = pc[:, 2] > 0.5 / ego.mpu
        u = np.where(front, fx * pc[:, 0] / np.maximum(pc[:, 2], 1e-9) + cx, -1)
        y = np.where(front, fy * pc[:, 1] / np.maximum(pc[:, 2], 1e-9) + cy, -1)
        inside = front & (u >= 0) & (u < w) & (y >= 0) & (y < h)
        if not inside.any():
            continue
        stem = Path(v.name).stem
        keep = _mask(dense / "images" / "masks" / f"{stem}.png", (w, h), True)
        keep &= ~_mask(dense / "sky" / f"{stem}.png", (w, h), False)
        idx = np.nonzero(inside)[0]
        ui, yi = u[idx].astype(int), y[idx].astype(int)
        idx = idx[keep[yi, ui]]
        dist = np.linalg.norm(grid[idx] - v.centre, axis=1)
        better = dist < best[idx]
        if not better.any():
            continue
        idx, dist = idx[better], dist[better]
        img = np.asarray(Image.open(_image_path(dense, v.name)).convert("RGB"))
        rgb[idx] = img[y[idx].astype(int), u[idx].astype(int)]
        best[idx] = dist
    seen = np.isfinite(best)
    return grid[seen], rgb[seen]


def append_points(points_txt: Path, xyz: np.ndarray, rgb: np.ndarray) -> int:
    """Append points with fresh ids, error 0 and an empty track. No blank lines (Brush rejects them)."""
    lines = [line for line in points_txt.read_text().splitlines() if line.strip()]
    ids = [int(line.split()[0]) for line in lines if not line.startswith("#")]
    start = max(ids, default=0) + 1
    new = [
        f"{start + i} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0"
        for i, ((x, y, z), (r, g, b)) in enumerate(zip(xyz.tolist(), rgb.tolist(), strict=True))
    ]
    points_txt.write_text("\n".join(lines + new) + "\n")
    return len(new)


def run(work: Path, dense: Path, depth_fn=None, max_points: int = MAX_POINTS, seed: int = 0) -> dict:
    """Add dense depth points and the road ribbon to dense/sparse/points3D.txt. Returns counts."""
    sparse = dense / "sparse"
    cams, views, pts = read_model(sparse)
    ego = EgoPath.load(work / "viewer" / "path.json")
    depth_fn = depth_fn or DepthModel()
    xyz, rgb = backproject(dense, cams, views, pts, depth_fn, MAX_DEPTH_M / ego.mpu)
    raw = len(xyz)
    if raw:
        lateral, _, _ = ego.nearest_road(xyz)
        voxel = np.where(lateral <= NEAR_ROAD_M, VOXEL_NEAR_M, VOXEL_FAR_M) / ego.mpu
        xyz, rgb = voxel_downsample(xyz, rgb, voxel)
    if len(xyz) > max_points:
        keep = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    rx, rc = road_ribbon(dense, cams, views, ego)
    added = append_points(sparse / "points3D.txt", np.concatenate([xyz, rx]), np.concatenate([rgb, rc]))
    counts = {"sfm": len(pts), "depth_raw": raw, "depth": len(xyz), "ribbon": len(rx), "added": added}
    print(f"dense init: +{len(xyz)} depth points (from {raw}), +{len(rx)} road-ribbon points (SfM had {len(pts)})")
    return counts

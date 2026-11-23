"""Gaussian-splat PLY I/O and the ego-path geometry shared by bench, prune and collision.

Splat PLYs (as Brush exports them) are one `vertex` element of scalar properties: x y z, opacity (a
logit), scale_0..2 (log scale), rot_0..3, f_dc_*, f_rest_*. `read_ply` keeps every property in file
order; `write_ply` writes them all back as binary little-endian float32, so a pruned scene keeps
whatever the trainer exported (SH degree, normals, extra fields).

Scene space is COLMAP's world frame in arbitrary units. viewer/path.json gives `up` (unit vector),
`metresPerUnit` and the front camera's smoothed path `frames[].p`; the road is CAM_HEIGHT_M below it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

CAM_HEIGHT_M = 1.4  # front camera above the road (same constant as the viewer and nvs-export)
# The ego swept tube: the car's own volume along the path. Splats in it are floaters by definition.
TUBE_RADIUS_M = 1.0
TUBE_LO_M = 0.15
TUBE_HI_M = 1.6

_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass
class Ply:
    data: np.ndarray  # structured array, one field per vertex property, in file order
    comments: list[str]

    def __len__(self) -> int:
        return len(self.data)

    def xyz(self) -> np.ndarray:
        return np.stack([self.data["x"], self.data["y"], self.data["z"]], 1).astype(np.float64)

    def opacity(self) -> np.ndarray:
        """Opacity in [0, 1] (the file stores a logit)."""
        return 1.0 / (1.0 + np.exp(-self.data["opacity"].astype(np.float64)))

    def max_scale(self) -> np.ndarray:
        """Largest axis of each splat, in scene units (the file stores log scales)."""
        names = [n for n in ("scale_0", "scale_1", "scale_2") if n in self.data.dtype.names]
        if not names:
            return np.zeros(len(self))
        return np.exp(np.max(np.stack([self.data[n] for n in names], 1).astype(np.float64), 1))

    def subset(self, keep: np.ndarray) -> Ply:
        return Ply(self.data[keep], list(self.comments))


def read_ply(path: str | Path) -> Ply:
    raw = Path(path).read_bytes()
    end = raw.find(b"end_header")
    if not raw.startswith(b"ply") or end < 0:
        raise ValueError(f"{path}: not a PLY file")
    body = raw.index(b"\n", end) + 1
    header = raw[:body].decode("ascii", "replace").splitlines()
    fmt, comments, fields, count, element = None, [], [], 0, None
    for line in header[1:]:
        p = line.split()
        if not p:
            continue
        if p[0] == "format":
            fmt = p[1]
        elif p[0] == "comment":
            comments.append(line[len("comment ") :])
        elif p[0] == "element":
            if element == "vertex":
                break  # anything after the vertex element is not ours to read
            element = p[1]
            if element == "vertex":
                count = int(p[2])
        elif p[0] == "property" and element == "vertex":
            if p[1] == "list":
                raise ValueError(f"{path}: list properties are not supported in splat PLYs")
            fields.append((p[2], _PLY_TYPES[p[1]]))
    if fmt != "binary_little_endian":
        raise ValueError(f"{path}: only binary_little_endian PLYs are supported (got {fmt})")
    if element is None or not fields:
        raise ValueError(f"{path}: no vertex element")
    dt = np.dtype([(n, "<" + t) for n, t in fields])
    data = np.frombuffer(raw, dtype=dt, count=count, offset=body).copy()
    return Ply(data, comments)


def write_ply(path: str | Path, ply: Ply) -> None:
    """All properties as little-endian float32, in the original order."""
    names = list(ply.data.dtype.names)
    out = np.empty(len(ply.data), dtype=[(n, "<f4") for n in names])
    for n in names:
        out[n] = ply.data[n]
    head = ["ply", "format binary_little_endian 1.0", *(f"comment {c}" for c in ply.comments)]
    head += [f"element vertex {len(out)}", *(f"property float {n}" for n in names), "end_header"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(out.tobytes())


@dataclass
class EgoPath:
    """The ego car's path in scene units, plus the frame needed to talk about it in metres."""

    P: np.ndarray  # (n, 3) front-camera positions
    up: np.ndarray  # (3,) unit
    mpu: float  # metres per scene unit

    @classmethod
    def load(cls, path_json: str | Path) -> EgoPath:
        return cls.from_dict(json.loads(Path(path_json).read_text()))

    @classmethod
    def from_dict(cls, d: dict) -> EgoPath:
        up = np.asarray(d["up"], float)
        P = np.asarray([f["p"] for f in d["frames"]], float)
        if len(P) < 2:
            raise ValueError("path.json needs at least 2 frames")
        return cls(P, up / np.linalg.norm(up), float(d.get("metresPerUnit") or 1.0))

    @property
    def road(self) -> np.ndarray:
        return self.P - self.up * (CAM_HEIGHT_M / self.mpu)

    def horizontal_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """(right, fwd): unit vectors perpendicular to `up`; fwd is the path's mean direction."""
        d = self.P[-1] - self.P[0]
        d = d - (d @ self.up) * self.up
        if np.linalg.norm(d) < 1e-9:  # degenerate (parked): any horizontal direction
            d = np.cross(self.up, [1.0, 0, 0] if abs(self.up[0]) < 0.9 else [0, 0, 1.0])
        fwd = d / np.linalg.norm(d)
        right = np.cross(fwd, self.up)
        return right / np.linalg.norm(right), fwd

    def dense_road(self, step_m: float = 0.1) -> np.ndarray:
        """Road points along the path, resampled every `step_m` metres (linear between frames)."""
        R = self.road
        seg = np.linalg.norm(np.diff(R, axis=0), axis=1) * self.mpu
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] <= 0:
            return R[:1]
        q = np.arange(0.0, s[-1] + 1e-9, step_m)
        return np.stack([np.interp(q, s, R[:, k]) for k in range(3)], 1)

    def nearest_road(self, xyz: np.ndarray, step_m: float = 0.1):
        """Per point: horizontal (lateral) distance to the path in metres, height above the nearest
        road point in metres, and that road point's index into dense_road(step_m)."""
        road = self.dense_road(step_m)
        right, fwd = self.horizontal_basis()
        H = np.stack([right, fwd], 1)
        _, idx = cKDTree(road @ H).query(np.asarray(xyz) @ H, workers=-1)
        rel = np.asarray(xyz) - road[idx]
        lateral = np.linalg.norm(rel @ H, axis=1) * self.mpu
        height = (rel @ self.up) * self.mpu
        return lateral, height, idx

    def tube_mask(
        self, xyz: np.ndarray, radius_m: float = TUBE_RADIUS_M, lo_m: float = TUBE_LO_M, hi_m: float = TUBE_HI_M
    ) -> np.ndarray:
        """True for points inside the ego swept tube (within radius_m of the path laterally and
        lo_m..hi_m above the road)."""
        if len(xyz) == 0:
            return np.zeros(0, bool)
        lateral, height, _ = self.nearest_road(xyz)
        return (lateral <= radius_m) & (height >= lo_m) & (height <= hi_m)


def floaters(ply: Ply, ego: EgoPath, min_opacity: float = 0.1) -> int:
    """Splats with opacity > min_opacity inside the ego swept tube."""
    return int((ego.tube_mask(ply.xyz()) & (ply.opacity() > min_opacity)).sum())


def prune(ply: Ply, ego: EgoPath, giant_m: float = 3.0, giant_max_opacity: float = 0.2) -> tuple[Ply, dict[str, int]]:
    """Drop every splat inside the ego tube, and giant faint ones (any axis > giant_m metres with
    opacity < giant_max_opacity): the haze that makes renders look flat. Returns (kept, counts)."""
    tube = ego.tube_mask(ply.xyz())
    giant = (ply.max_scale() * ego.mpu > giant_m) & (ply.opacity() < giant_max_opacity)
    drop = tube | giant
    counts = {
        "before": len(ply),
        "tube": int(tube.sum()),
        "giant": int((giant & ~tube).sum()),
        "after": int((~drop).sum()),
    }
    return ply.subset(~drop), counts

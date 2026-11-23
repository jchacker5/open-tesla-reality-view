"""Synthetic splat scenes for tests: a straight drive along +z with up = -y (COLMAP-like)."""

import json

import numpy as np

from tsplat import splats

UP = np.array([0.0, -1.0, 0.0])


def logit(p):
    return float(np.log(p / (1 - p)))


def make_ply(xyz, opacity=0.9, scale_m=0.05, mpu=1.0, extra=True) -> splats.Ply:
    """Brush-like vertex layout (alphabetical property order, SH degree 1) with the given positions."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    n = len(xyz)
    names = ["f_dc_0", "f_dc_1", "f_dc_2"] + ([f"f_rest_{i}" for i in range(9)] if extra else [])
    names += ["opacity", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2", "x", "y", "z"]
    data = np.zeros(n, dtype=[(k, "<f4") for k in names])
    data["x"], data["y"], data["z"] = xyz.T
    data["opacity"] = np.broadcast_to(np.vectorize(logit)(np.asarray(opacity, float)), n)
    for k in ("scale_0", "scale_1", "scale_2"):
        data[k] = np.log(np.broadcast_to(np.asarray(scale_m, float) / mpu, n))
    data["rot_0"] = 1.0
    data["f_dc_0"] = np.arange(n) % 7  # something to check survives a round trip
    return splats.Ply(data, ["Exported from Brush", "Vertical axis: y"])


def path_dict(n=21, step=1.0, mpu=1.0, cam_h=1.4):
    """Front camera at height cam_h (metres) above the road plane y = 0, driving along +z."""
    frames = [{"t": i * 0.5, "p": [0.0, -cam_h / mpu, i * step], "f": [0, 0, 1]} for i in range(n)]
    return {"up": UP.tolist(), "metresPerUnit": mpu, "frames": frames}


def write_path(path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(path_dict(**kw)))
    return path

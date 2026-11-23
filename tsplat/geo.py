"""Align a COLMAP trajectory to Tesla GPS fixes (WGS84)."""

from __future__ import annotations

import numpy as np

EARTH_M = 6378137.0


def align_track(frames, up, metres_per_unit: float, telemetry) -> dict | None:
    """Georeference SLAM frames with lat/lon samples. None without two frames and two usable fixes.

    A fix is usable when lat/lon are finite, inside ±90/±180, and not null island.
    Metres are scene units times metres_per_unit. A GPS span under 8 m cannot support a
    scale, so the fit stays rigid and trusts metres_per_unit.
    """
    if not frames or len(frames) < 2:
        return None
    fixes = _usable_fixes(telemetry)
    if len(fixes) < 2:
        return None

    up = np.asarray(up, float)
    up_n = float(np.linalg.norm(up))
    if up_n < 1e-12:
        return None
    up = up / up_n
    right, fwd = _ground_axes(np.asarray(frames[0]["f"], float), up)
    if right is None:
        return None

    origin = np.asarray(frames[0]["p"], float)
    m = float(metres_per_unit)

    def xy_of(points: np.ndarray) -> np.ndarray:
        delta = (np.asarray(points, float) - origin) * m
        return np.column_stack((delta @ right, delta @ fwd))

    times = np.array([fix[0] for fix in fixes], float)
    src = xy_of(_positions_at(frames, times))
    lat0, lon0 = fixes[0][1], fixes[0][2]
    lats = np.array([fix[1] for fix in fixes], float)
    lons = np.array([fix[2] for fix in fixes], float)
    north = np.radians(lats - lat0) * EARTH_M
    east = np.radians(lons - lon0) * EARTH_M * np.cos(np.radians(lat0))
    dst = np.column_stack((east, north))

    scale, rotation, translation = _similarity(src, dst)
    if _max_span(dst) < 8.0:
        scale = 1.0
        translation = dst.mean(axis=0) - rotation @ src.mean(axis=0)

    xy = xy_of(np.vstack([np.asarray(f["p"], float) for f in frames]))
    enu = scale * (xy @ rotation.T) + translation
    lat = lat0 + np.degrees(enu[:, 1] / EARTH_M)
    lon = lon0 + np.degrees(enu[:, 0] / (EARTH_M * np.cos(np.radians(lat0))))
    pred = scale * (src @ rotation.T) + translation
    rms = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return {
        "origin": {"lat": float(lat0), "lon": float(lon0)},
        "rms_m": round(rms, 3),
        "fixes": len(fixes),
        "scale": round(float(scale), 4),
        "frames": [
            {"t": frame["t"], "lat": round(float(lat[i]), 7), "lon": round(float(lon[i]), 7)}
            for i, frame in enumerate(frames)
        ],
    }


def _usable_fixes(telemetry) -> list[tuple[float, float, float]]:
    fixes: list[tuple[float, float, float]] = []
    if not telemetry:
        return fixes
    for sample in telemetry:
        if not isinstance(sample, dict):
            continue
        lat, lon, t = sample.get("lat"), sample.get("lon"), sample.get("t")
        if lat is None or lon is None or t is None:
            continue
        try:
            lat_f, lon_f, t_f = float(lat), float(lon), float(t)
        except (TypeError, ValueError):
            continue
        if not np.isfinite([lat_f, lon_f, t_f]).all():
            continue
        if abs(lat_f) > 90.0 or abs(lon_f) > 180.0:
            continue
        if abs(lat_f) < 1e-7 and abs(lon_f) < 1e-7:
            continue
        fixes.append((t_f, lat_f, lon_f))
    return fixes


def _ground_axes(forward: np.ndarray, up: np.ndarray):
    horizontal = forward - np.dot(forward, up) * up
    if float(np.linalg.norm(horizontal)) < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
        horizontal = axis - np.dot(axis, up) * up
    if float(np.linalg.norm(horizontal)) < 1e-8:
        return None, None
    right = np.cross(horizontal, up)
    right_n = float(np.linalg.norm(right))
    if right_n < 1e-8:
        return None, None
    right = right / right_n
    fwd = np.cross(up, right)
    fwd = fwd / np.linalg.norm(fwd)
    return right, fwd


def _positions_at(frames, times: np.ndarray) -> np.ndarray:
    ts = np.array([float(f["t"]) for f in frames], float)
    positions = np.vstack([np.asarray(f["p"], float) for f in frames])
    order = np.argsort(ts, kind="stable")
    ts, positions = ts[order], positions[order]
    keep = np.ones(len(ts), dtype=bool)
    keep[1:] = np.diff(ts) > 0
    ts, positions = ts[keep], positions[keep]
    return np.column_stack([np.interp(times, ts, positions[:, k]) for k in range(3)])


def _max_span(points: np.ndarray) -> float:
    delta = points[:, None, :] - points[None, :, :]
    return float(np.sqrt((delta * delta).sum(axis=-1)).max())


def _similarity(src: np.ndarray, dst: np.ndarray):
    """2D Umeyama, column vectors: dst ≈ scale * R @ src + t."""
    n = src.shape[0]
    mu_x = src.mean(axis=0)
    mu_y = dst.mean(axis=0)
    centred_x = src - mu_x
    centred_y = dst - mu_y
    sigma = (centred_y.T @ centred_x) / n
    u, singular, vt = np.linalg.svd(sigma)
    fix = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        fix[1, 1] = -1.0
    rotation = u @ fix @ vt
    var_x = float((centred_x * centred_x).sum() / n)
    if var_x < 1e-12:
        scale = 1.0
    else:
        scale = float(np.trace(np.diag(singular) @ fix) / var_x)
    translation = mu_y - scale * (rotation @ mu_x)
    return scale, rotation, translation

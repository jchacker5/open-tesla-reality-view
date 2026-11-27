"""SEI GPS -> local ENU coordinates and COLMAP 4 pose_priors table.

COLMAP 4.x `pose_prior_mapper` reads camera-centre priors from the `pose_priors` table in
`colmap.db` (PosePrior::CoordinateSystem::CARTESIAN = 1, SensorType::CAMERA = 0). Feeding
per-frame GPS from Tesla's SEI telemetry pins the reconstruction to metric ENU scale and reduces
drift on longer windows.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np

# WGS84 constants (match cpp/src/geo.cpp).
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)

SENSOR_TYPE_CAMERA = 0
COORD_SYSTEM_CARTESIAN = 1


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    """Convert WGS84 geodetic degrees to ECEF metres."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sl * sl)
    return np.array(
        [(n + alt_m) * cl * co, (n + alt_m) * cl * so, (n * (1.0 - _WGS84_E2) + alt_m) * sl],
        dtype=np.float64,
    )


def geodetic_to_enu(
    lat_deg: float, lon_deg: float, origin_lat: float, origin_lon: float, alt_m: float = 0.0
) -> np.ndarray:
    """Convert WGS84 (lat, lon, alt) to local East-North-Up metres around `(origin_lat, origin_lon)`."""
    lat0 = np.radians(origin_lat)
    lon0 = np.radians(origin_lon)
    sl, cl = np.sin(lat0), np.cos(lat0)
    so, co = np.sin(lon0), np.cos(lon0)
    basis = np.array(
        [[-so, co, 0.0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]],
        dtype=np.float64,
    )
    return basis @ (geodetic_to_ecef(lat_deg, lon_deg, alt_m) - geodetic_to_ecef(origin_lat, origin_lon, 0.0))


def clean_gps_fixes(tel: list[dict]) -> list[tuple[float, float, float]]:
    """Return time-sorted `(t, lat, lon)` fixes without (0, 0) placeholders or duplicate holds."""
    rows: list[tuple[float, float, float]] = []
    for s in tel:
        t, lat, lon = s.get("t"), s.get("lat"), s.get("lon")
        if t is None or lat is None or lon is None:
            continue
        tf, latf, lonf = float(t), float(lat), float(lon)
        if not (np.isfinite(tf) and np.isfinite(latf) and np.isfinite(lonf)):
            continue
        if latf == 0.0 and lonf == 0.0:
            continue
        rows.append((tf, latf, lonf))
    rows.sort(key=lambda r: r[0])
    fixes: list[tuple[float, float, float]] = []
    for t, lat, lon in rows:
        if fixes and fixes[-1][1] == lat and fixes[-1][2] == lon:
            continue
        fixes.append((t, lat, lon))
    return fixes


def sample_frame_enu(
    tel: list[dict],
    frame_times: dict[int, float],
    max_gap_s: float = 3.0,
    min_extent_m: float = 2.0,
) -> tuple[dict[int, np.ndarray], tuple[float, float]]:
    """Interpolate local ENU positions (metres) at each frame timestamp from SEI telemetry."""
    fixes = clean_gps_fixes(tel)
    if len(fixes) < 2:
        raise ValueError("fewer than two distinct GPS fixes in telemetry")
    origin_lat, origin_lon = fixes[0][1], fixes[0][2]
    fix_ts = np.array([f[0] for f in fixes], dtype=np.float64)
    fix_enu = np.stack([geodetic_to_enu(f[1], f[2], origin_lat, origin_lon) for f in fixes], axis=0)
    extent = float(np.hypot(np.ptp(fix_enu[:, 0]), np.ptp(fix_enu[:, 1])))
    if extent < min_extent_m:
        raise ValueError(f"GPS path extent ({extent:.1f} m) is too small for pose priors")

    out: dict[int, np.ndarray] = {}
    for idx, t in frame_times.items():
        if t < fix_ts[0] or t > fix_ts[-1]:
            continue
        hi = int(np.searchsorted(fix_ts, t, side="left"))
        if fix_ts[hi] == t:
            out[idx] = fix_enu[hi].copy()
            continue
        if hi == 0 or fix_ts[hi] - fix_ts[hi - 1] > max_gap_s:
            continue
        w = (t - fix_ts[hi - 1]) / (fix_ts[hi] - fix_ts[hi - 1])
        out[idx] = fix_enu[hi - 1] + w * (fix_enu[hi] - fix_enu[hi - 1])
    if len(out) < 2:
        raise ValueError("fewer than two sampled frames fall inside usable GPS fixes")
    return out, (origin_lat, origin_lon)


def write_pose_priors(
    db_path: Path,
    frame_enu: dict[int, np.ndarray],
    sigma_h_m: float = 4.0,
    sigma_v_m: float = 10.0,
    ref_cam: str = "front",
) -> int:
    """Populate COLMAP 4's `pose_priors` table for reference-camera images matching `frame_enu`."""
    if sigma_h_m <= 0 or sigma_v_m <= 0:
        raise ValueError("GPS prior standard deviations must be positive")
    cov = struct.pack(
        "<9d",
        sigma_h_m**2, 0.0, 0.0,
        0.0, sigma_h_m**2, 0.0,
        0.0, 0.0, sigma_v_m**2,
    )
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id").fetchall()
        prefix = f"{ref_cam}/"
        selected = [r for r in rows if r[1].startswith(prefix)] or rows
        conn.execute("DELETE FROM pose_priors")
        inserted = 0
        for image_id, name, camera_id in selected:
            stem = Path(name).stem.rsplit("_", 1)[-1]
            try:
                idx = int(stem)
            except ValueError:
                continue
            pos = frame_enu.get(idx)
            if pos is None:
                continue
            inserted += 1
            conn.execute(
                "INSERT INTO pose_priors "
                "(pose_prior_id, corr_data_id, corr_sensor_id, corr_sensor_type, "
                "position, position_covariance, gravity, coordinate_system) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    inserted,
                    image_id,
                    camera_id,
                    SENSOR_TYPE_CAMERA,
                    struct.pack("<3d", float(pos[0]), float(pos[1]), float(pos[2])),
                    cov,
                    COORD_SYSTEM_CARTESIAN,
                ),
            )
        conn.commit()
        return inserted
    finally:
        conn.close()

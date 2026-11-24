"""Put a reconstruction on the map with the car's own GPS (C++: cpp/src/geo.cpp).

    viewer/path.json + viewer/telemetry.json -> viewer/georef.json

SfM recovers the drive up to an unknown scale, rotation and offset. The SEI telemetry has the
car's latitude and longitude in every frame, so fitting the path to the GPS track gives metric
scale, true north and a geographic origin at once. The fit's scale is an independent check on the
speed-derived `metresPerUnit` that the viewer uses: the two should agree within a few percent.
"""

from __future__ import annotations

import json
from pathlib import Path

# Above this the speed and GPS scales disagree enough to suggest a bad reconstruction or bad GPS.
SCALE_WARN = 0.10


def available() -> bool:
    try:
        import otrv_native  # noqa: F401
    except ImportError:
        return False
    return True


def inputs(work: Path) -> tuple[list, list, list]:
    path = json.loads((work / "viewer" / "path.json").read_text())
    tel = json.loads((work / "viewer" / "telemetry.json").read_text())
    frames = [(f["t"], *f["p"]) for f in path["frames"]]
    gps = [(s["t"], s["lat"], s["lon"]) for s in tel if s.get("lat") is not None and s.get("lon") is not None]
    return frames, path["up"], gps


def run(work: Path, inlier_m: float = 3.0, quiet: bool = False) -> dict:
    """Fit, write viewer/georef.json (also on failure, with the reason) and return the result."""
    if not available():
        raise SystemExit("otrv georef needs the C++ core: `uv sync --extra native`")
    import otrv_native

    frames, up, gps = inputs(work)
    res = otrv_native.geo_fit(frames, up, gps, inlier_m=inlier_m)
    path = json.loads((work / "viewer" / "path.json").read_text())
    speed_scale = path.get("metresPerUnit")
    if res["ok"] and speed_scale:
        res["speed_metres_per_unit"] = speed_scale
        res["scale_ratio"] = res["metres_per_unit"] / speed_scale
    (work / "viewer" / "georef.json").write_text(json.dumps(res))
    if not quiet:
        print(summary(res))
    return res


def summary(res: dict) -> str:
    if not res["ok"]:
        return f"georef: no fit ({res['reason']}; {res['matched']} path samples had GPS)"
    lat, lon = res["origin"]
    line = (
        f"georef: {res['inliers']}/{res['matched']} samples agree with GPS, rms {res['rms_m']:.2f} m, "
        f"{res['metres_per_unit']:.3f} m/unit, heading offset {res['yaw_deg']:.1f} deg, origin {lat:.6f}, {lon:.6f}"
    )
    if "scale_ratio" in res:
        r = res["scale_ratio"]
        line += f"\n        GPS scale / speed scale = {r:.3f}"
        if abs(r - 1) > SCALE_WARN:
            line += f" (off by {abs(r - 1):.0%}: check the reconstruction or the GPS)"
    return line

"""otrv georef end to end on a synthetic run: a reconstruction under an arbitrary similarity, and the
car's GPS at 10 Hz with receiver noise and multipath jumps."""

import json
import math
import os

import numpy as np
import pytest

from tsplat import georef

if not georef.available():
    if os.environ.get("OTRV_REQUIRE_NATIVE") == "1":
        raise RuntimeError("OTRV_REQUIRE_NATIVE=1 but otrv_native is not importable")
    pytest.skip("otrv_native not installed (uv sync --extra native)", allow_module_level=True)

LAT0, LON0 = 41.7015, -71.155
A, E2 = 6378137.0, 6.69437999014e-3


def enu_to_latlon(e, n):
    """Local tangent-plane inverse with WGS84 radii of curvature: centimetre-accurate over a few hundred metres."""
    s = math.sin(math.radians(LAT0))
    m = A * (1 - E2) / (1 - E2 * s * s) ** 1.5
    nn = A / math.sqrt(1 - E2 * s * s)
    return LAT0 + math.degrees(n / m), LON0 + math.degrees(e / (nn * math.cos(math.radians(LAT0))))


def drive(t):
    """12.5 m/s: 8 s north-east, then a left turn of radius 25 m."""
    v, h = 12.5, 0.35
    if t <= 8:
        return np.array([v * t * math.cos(h), v * t * math.sin(h)])
    c0 = np.array([v * 8 * math.cos(h), v * 8 * math.sin(h)])
    centre = c0 + 25 * np.array([-math.sin(h), math.cos(h)])
    a = h - math.pi / 2 + min((t - 8) * v / 25, math.pi / 2)
    return centre + 25 * np.array([math.cos(a), math.sin(a)])


def make_run(tmp_path, gps=True, speed_scale_error=1.03):
    units_per_m = 0.42
    yaw = 1.2
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    # SfM's frame: y down is "up" negated, like COLMAP's camera convention
    flip = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r = flip @ rz
    offset = np.array([3.0, 1.0, -2.0])
    frames = []
    for i in range(0, 15 * 6 + 1):
        t = i / 6
        e, n = drive(t)
        p = units_per_m * (r @ np.array([e, n, 0.0])) + offset
        frames.append({"t": round(t, 3), "p": p.round(5).tolist(), "f": [1, 0, 0]})
    rng = np.random.default_rng(0)
    tel = []
    for i in range(0, 15 * 10 + 1):
        t = i / 10
        e, n = drive(t) + rng.normal(0, 0.8, 2)
        if i % 17 == 5:
            e, n = e + 35, n - 20  # multipath
        lat, lon = enu_to_latlon(e, n) if gps else (None, None)
        tel.append({"t": t, "mph": 28.0, "lat": lat, "lon": lon})
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    up = (r @ np.array([0, 0, 1.0])).tolist()
    (viewer / "path.json").write_text(
        json.dumps({"up": up, "metresPerUnit": speed_scale_error / units_per_m, "frames": frames})
    )
    (viewer / "telemetry.json").write_text(json.dumps(tel))
    return 1 / units_per_m


def test_georef_recovers_scale_heading_and_origin(tmp_path, capsys):
    true_scale = make_run(tmp_path)
    res = georef.run(tmp_path)
    assert res["ok"], res["reason"]
    assert res["metres_per_unit"] == pytest.approx(true_scale, rel=0.01)
    assert res["scale_ratio"] == pytest.approx(1 / 1.03, rel=0.01)
    assert res["rms_m"] < 1.5 and res["inliers"] < res["matched"]  # the multipath samples were dropped
    lat, lon = res["track"][0][1:]
    assert lat == pytest.approx(LAT0, abs=2e-5) and lon == pytest.approx(LON0, abs=2e-5)  # ~2 m
    # the end of the turn lands where the car really was
    e, n = drive(15)
    want = enu_to_latlon(e, n)
    assert res["track"][-1][1] == pytest.approx(want[0], abs=2e-5)
    assert res["track"][-1][2] == pytest.approx(want[1], abs=2e-5)
    saved = json.loads((tmp_path / "viewer" / "georef.json").read_text())
    assert saved["ok"] and len(saved["matrix"]) == 16
    out = capsys.readouterr().out
    assert "GPS scale / speed scale = 0.9" in out and "off by" not in out


def test_georef_flags_a_scale_disagreement(tmp_path):
    make_run(tmp_path, speed_scale_error=1.25)
    res = georef.run(tmp_path, quiet=True)
    assert res["ok"]
    assert "off by 20%" in georef.summary(res)


def test_georef_without_gps_reports_why(tmp_path, capsys):
    make_run(tmp_path, gps=False)
    res = georef.run(tmp_path)
    assert not res["ok"] and "GPS fixes" in res["reason"]
    assert json.loads((tmp_path / "viewer" / "georef.json").read_text())["ok"] is False
    assert "no fit" in capsys.readouterr().out

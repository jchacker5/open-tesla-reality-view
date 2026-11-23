import numpy as np
import pytest

from tsplat.geo import EARTH_M, align_track

LAT0 = 41.7012
LON0 = -71.1553
UP = [0.0, 1.0, 0.0]


def ll_of(east: float, north: float, lat0: float = LAT0, lon0: float = LON0):
    lat = lat0 + np.degrees(north / EARTH_M)
    lon = lon0 + np.degrees(east / (EARTH_M * np.cos(np.radians(lat0))))
    return float(lat), float(lon)


def enu_of(lat: float, lon: float, lat0: float, lon0: float):
    north = np.radians(lat - lat0) * EARTH_M
    east = np.radians(lon - lon0) * EARTH_M * np.cos(np.radians(lat0))
    return float(east), float(north)


def frames_on_ground(samples, metres_per_unit: float = 1.0):
    """(t, right_m, fwd_m) with up = +y and forward = +z, so xy = (right_m, fwd_m)."""
    frames = []
    for t, right_m, fwd_m in samples:
        frames.append(
            {
                "t": t,
                "p": [-right_m / metres_per_unit, 0.0, fwd_m / metres_per_unit],
                "f": [0.0, 0.0, 1.0],
            }
        )
    return frames


def test_rotated_drive_matches_gps():
    """Straight +z drive, 1 scene unit/s, 10 m/unit, GPS rotated 30 deg off 41.7012, -71.1553."""
    metres_per_unit = 10.0
    theta = np.radians(30.0)
    frames = []
    telemetry = []
    expected = []
    for i in range(21):  # 0..20 s → 200 m
        frames.append({"t": float(i), "p": [0.0, 0.0, float(i)], "f": [0.0, 0.0, 1.0]})
        distance = i * metres_per_unit
        east = -distance * np.sin(theta)
        north = distance * np.cos(theta)
        lat, lon = ll_of(east, north)
        telemetry.append({"t": float(i), "lat": lat, "lon": lon})
        expected.append((lat, lon))

    result = align_track(frames, UP, metres_per_unit, telemetry)
    assert result is not None
    assert result["rms_m"] < 0.05
    assert result["scale"] == pytest.approx(1.0, abs=1e-3)
    assert result["fixes"] == 21
    assert result["origin"]["lat"] == pytest.approx(LAT0, abs=1e-12)
    assert result["origin"]["lon"] == pytest.approx(LON0, abs=1e-12)
    assert [f["t"] for f in result["frames"]] == [f["t"] for f in frames]
    for got, (lat, lon) in zip(result["frames"], expected, strict=True):
        assert got["lat"] == pytest.approx(lat, abs=1e-5)
        assert got["lon"] == pytest.approx(lon, abs=1e-5)


def test_two_fixes_cover_every_frame():
    frames = [{"t": float(i), "p": [0.0, 0.0, float(i)], "f": [0.0, 0.0, 1.0]} for i in range(6)]
    telemetry = [
        {"t": 0.0, "lat": LAT0, "lon": LON0},
        {"t": 5.0, "lat": LAT0 + 0.001, "lon": LON0 + 0.001},
    ]
    result = align_track(frames, UP, 10.0, telemetry)
    assert result is not None
    assert result["fixes"] == 2
    assert len(result["frames"]) == len(frames)
    assert [f["t"] for f in result["frames"]] == [f["t"] for f in frames]


def test_too_few_usable_fixes_return_none():
    frames = [
        {"t": 0.0, "p": [0.0, 0.0, 0.0], "f": [0.0, 0.0, 1.0]},
        {"t": 1.0, "p": [0.0, 0.0, 1.0], "f": [0.0, 0.0, 1.0]},
    ]
    real = {"t": 0.0, "lat": LAT0, "lon": LON0}
    other = {"t": 1.0, "lat": LAT0 + 0.01, "lon": LON0 + 0.01}
    zero = {"t": 1.0, "lat": 0.0, "lon": 0.0}
    assert align_track(frames, UP, 10.0, [real]) is None
    assert align_track(frames, UP, 10.0, [zero]) is None
    assert align_track(frames, UP, 10.0, []) is None
    assert align_track(frames, UP, 10.0, [real, zero]) is None
    assert align_track(frames, UP, 10.0, [zero, real]) is None
    assert align_track(frames, UP, 10.0, [real, {"t": 1.0, "lat": 1e-8, "lon": -1e-8}]) is None
    assert align_track(frames[:1], UP, 10.0, [real, other]) is None


def test_short_gps_span_forces_unit_scale():
    """Natural similarity is 1.15, but the GPS cloud is under 8 m on a long slam path."""
    triangle = [(0.0, 0.0), (5.0, 0.2), (1.0, 4.0)]
    factor = 1.15
    samples = [(float(i), x, y) for i, (x, y) in enumerate(triangle)]
    samples += [(float(i), 1.0, 4.0 + 20.0 * (i - 2)) for i in range(3, 19)]  # out to ~324 m
    frames = frames_on_ground(samples, 1.0)
    telemetry = []
    gps = []
    for t, x, y in samples[:3]:
        east, north = factor * x, factor * y
        lat, lon = ll_of(east, north)
        telemetry.append({"t": t, "lat": lat, "lon": lon})
        gps.append((lat, lon))

    result = align_track(frames, UP, 1.0, telemetry)
    assert result is not None
    assert result["scale"] == 1
    origin = result["origin"]
    for (lat, lon), got in zip(gps, result["frames"][:3], strict=True):
        east, north = enu_of(got["lat"], got["lon"], origin["lat"], origin["lon"])
        gps_east, gps_north = enu_of(lat, lon, origin["lat"], origin["lon"])
        assert np.hypot(east - gps_east, north - gps_north) < 2.0


def test_scale_two_and_quarter_turn():
    """metresPerUnit matches the scene; GPS is that geometry times 2, rotated 90 deg, span > 8 m."""
    xy = [(0.0, 0.0), (25.0, 0.0), (25.0, 30.0), (0.0, 30.0), (12.5, 0.0)]
    samples = [(float(i), x, y) for i, (x, y) in enumerate(xy)]
    frames = frames_on_ground(samples, 1.0)
    telemetry = []
    expected = []
    for t, x, y in samples:
        east, north = -2.0 * y, 2.0 * x  # scale 2, 90 deg CCW: (x, y) -> (-y, x)
        lat, lon = ll_of(east, north)
        telemetry.append({"t": t, "lat": lat, "lon": lon})
        expected.append((lat, lon))

    result = align_track(frames, UP, 1.0, telemetry)
    assert result is not None
    assert result["scale"] == pytest.approx(2.0, abs=1e-3)
    assert result["fixes"] == len(frames)
    for got, (lat, lon) in zip(result["frames"], expected, strict=True):
        assert got["lat"] == pytest.approx(lat, abs=1e-5)
        assert got["lon"] == pytest.approx(lon, abs=1e-5)

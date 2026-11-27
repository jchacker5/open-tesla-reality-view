import json
import sqlite3
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.test_recon import make_work
from tsplat import gps_priors, recon


def test_geodetic_to_enu_roundtrip_scale():
    # ~111 m per 0.001 deg latitude; ~79 m per 0.001 deg longitude near Boston (42.36 N).
    lat0, lon0 = 42.3601, -71.0589
    north = gps_priors.geodetic_to_enu(lat0 + 0.001, lon0, lat0, lon0)
    east = gps_priors.geodetic_to_enu(lat0, lon0 + 0.001, lat0, lon0)
    assert north[1] == pytest.approx(111.08, abs=0.2) and abs(north[0]) < 1e-6
    assert east[0] == pytest.approx(82.4, abs=0.4) and abs(east[1]) < 1e-3


def test_sample_frame_enu_filters_placeholders_and_interpolates():
    lat0, lon0 = 42.3601, -71.0589
    tel = [
        {"t": 0.0, "lat": 0.0, "lon": 0.0},
        {"t": 0.0, "lat": lat0, "lon": lon0},
        {"t": 1.0, "lat": lat0, "lon": lon0},  # duplicate hold
        {"t": 2.0, "lat": lat0 + 0.0002, "lon": lon0},
        {"t": 8.0, "lat": lat0 + 0.0008, "lon": lon0},  # > max_gap_s=3 gap from t=2
    ]
    pts, origin = gps_priors.sample_frame_enu(tel, {0: 0.0, 1: 1.0, 2: 2.0, 3: 5.0, 4: 8.0})
    assert origin == (lat0, lon0)
    assert set(pts) == {0, 1, 2, 4}  # t=5 is skipped because gap 2..8 > 3s
    assert np.allclose(pts[0], [0, 0, 0])
    assert pts[1][1] == pytest.approx(pts[2][1] * 0.5, rel=1e-4)


def test_sample_frame_enu_rejects_stationary_or_missing():
    with pytest.raises(ValueError, match="fewer than two"):
        gps_priors.sample_frame_enu([{"t": 0.0, "lat": 42.36, "lon": -71.05}], {0: 0.0, 1: 1.0})
    with pytest.raises(ValueError, match="too small"):
        gps_priors.sample_frame_enu(
            [{"t": 0.0, "lat": 42.36, "lon": -71.05}, {"t": 1.0, "lat": 42.360001, "lon": -71.05}],
            {0: 0.0, 1: 1.0},
        )


def _make_colmap_db(db: Path, names: list[str]) -> None:
    subprocess.run(["colmap", "database_creator", "--database_path", str(db)], check=True, capture_output=True)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) "
        "VALUES (1, 0, 100, 100, ?, 1)",
        (struct.pack("<3d", 100.0, 50.0, 50.0),),
    )
    for i, name in enumerate(names, 1):
        conn.execute("INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, 1)", (i, name))
    conn.commit()
    conn.close()


def test_write_pose_priors_writes_colmap_cartesian_blobs(tmp_path):
    db = tmp_path / "colmap.db"
    try:
        _make_colmap_db(db, ["front/00000.jpg", "front/00001.jpg", "back/00000.jpg"])
    except FileNotFoundError:
        pytest.skip("needs colmap binary")
    n = gps_priors.write_pose_priors(
        db,
        {0: np.array([1.0, 2.0, 0.0]), 1: np.array([3.0, 4.0, 0.0])},
        sigma_h_m=4.0,
        sigma_v_m=10.0,
    )
    assert n == 2
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT corr_data_id, corr_sensor_id, corr_sensor_type, position, "
        "position_covariance, coordinate_system FROM pose_priors ORDER BY pose_prior_id"
    ).fetchall()
    conn.close()
    assert len(rows) == 2
    assert rows[0][0] == 1 and rows[0][1] == 1 and rows[0][2] == 0 and rows[0][5] == 1
    assert struct.unpack("<3d", rows[0][3]) == pytest.approx((1.0, 2.0, 0.0))
    cov = struct.unpack("<9d", rows[1][4])
    assert cov[0] == pytest.approx(16.0) and cov[4] == pytest.approx(16.0) and cov[8] == pytest.approx(100.0)


def test_sfm_uses_pose_prior_mapper_when_gps_present(tmp_path, monkeypatch):
    work = tmp_path / "w"
    (work / "images" / "front").mkdir(parents=True)
    (work / "viewer").mkdir(parents=True)
    for i in range(4):
        (work / "images" / "front" / f"{i:05d}.jpg").write_bytes(b"")
    (work / "meta.json").write_text(json.dumps({"fps": 1.0, "cams": ["front"]}))
    lat0, lon0 = 42.3601, -71.0589
    (work / "viewer" / "telemetry.json").write_text(
        json.dumps([{"t": float(i), "lat": lat0 + i * 0.0001, "lon": lon0} for i in range(4)])
    )
    calls = []

    def fake_sh(*args):
        calls.append(args)
        if args[1] == "feature_extractor":
            db = Path(args[args.index("--database_path") + 1])
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
            conn.execute(
                "CREATE TABLE pose_priors (pose_prior_id INTEGER PRIMARY KEY, corr_data_id INTEGER, "
                "corr_sensor_id INTEGER, corr_sensor_type INTEGER, position BLOB, position_covariance BLOB, "
                "gravity BLOB, coordinate_system INTEGER)"
            )
            for i in range(4):
                conn.execute("INSERT INTO images VALUES (?, ?, 1)", (i + 1, f"front/{i:05d}.jpg"))
            conn.commit()
            conn.close()
        elif args[1] == "pose_prior_mapper":
            out = Path(args[args.index("--output_path") + 1]) / "0"
            out.mkdir(parents=True)
            (out / "images.bin").write_bytes(struct.pack("<Q", 4))

    monkeypatch.setattr(recon, "sh", fake_sh)
    model = recon.sfm(work, global_mapper=False, rig=False, gps_priors=True, gps_sigma_m=3.5)
    assert model == work / "sparse" / "0"
    assert any(c[1] == "pose_prior_mapper" for c in calls)
    meta = json.loads((work / "gps_priors.json").read_text())
    assert meta["priors"] == 4 and meta["sigma_h_m"] == 3.5


def test_sfm_falls_back_to_standard_mapper_without_gps(tmp_path, monkeypatch):
    work = tmp_path / "w"
    (work / "images" / "front").mkdir(parents=True)
    (work / "viewer").mkdir(parents=True)
    (work / "images" / "front" / "00000.jpg").write_bytes(b"")
    (work / "meta.json").write_text(json.dumps({"fps": 1.0, "cams": ["front"]}))
    (work / "viewer" / "telemetry.json").write_text("[]")
    calls = []

    def fake_sh(*args):
        calls.append(args)
        if args[1] == "mapper":
            out = Path(args[args.index("--output_path") + 1]) / "0"
            out.mkdir(parents=True)
            (out / "images.bin").write_bytes(struct.pack("<Q", 1))

    monkeypatch.setattr(recon, "sh", fake_sh)
    recon.sfm(work, global_mapper=False, rig=False, gps_priors=True)
    assert any(c[1] == "mapper" for c in calls)
    assert not (work / "gps_priors.json").exists()


def test_export_path_marks_gps_priors_scale(tmp_path):
    work = make_work(tmp_path)
    (work / "gps_priors.json").write_text(json.dumps({"origin": [42.36, -71.05], "priors": 6}))
    path = json.loads(recon.export_path(work, work / "dense").read_text())
    assert path["metresPerUnit"] == 1.0
    assert path["scaleSource"] == "gps-priors"

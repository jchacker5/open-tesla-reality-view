import json
import struct
import sys

import numpy as np
import pytest

from tsplat import bench, prep, recon


@pytest.mark.parametrize("q", [(1, 0, 0, 0), (0.3, -0.5, 0.7, 0.1), (0, 0, 1, 0)])
def test_qvec_to_R_orthonormal(q):
    q = np.array(q, float)
    R = recon.qvec_to_R(q / np.linalg.norm(q))
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.linalg.det(R) == pytest.approx(1.0)


def make_work(tmp_path, n=6, tel=None):
    """Front camera driving straight along +z, 1 unit per frame, identity rotation.
    Frame 2 has an empty POINTS2D line, as COLMAP writes for images without observations."""
    work = tmp_path / "w"
    sp = work / "dense" / "sparse"
    sp.mkdir(parents=True)
    (work / "viewer").mkdir()
    (work / "meta.json").write_text(
        json.dumps({"clip": "c", "timestamp": "t", "start": 0, "dur": n, "fps": 1.0, "cams": ["front"]})
    )
    (work / "viewer" / "telemetry.json").write_text(json.dumps(tel or []))
    (sp / "cameras.txt").write_text("# Camera list\n1 PINHOLE 640 480 500 500 320 240\n")
    lines = ["# Image list", "# IMAGE_ID ..."]
    for i in range(n):
        lines.append(f"{i + 1} 1 0 0 0 0 0 {-float(i)} 1 front/front_{i:05d}.jpg")
        lines.append("" if i == 2 else "10.0 20.0 -1 30.0 40.0 -1")
    (sp / "images.txt").write_text("\n".join(lines) + "\n")
    return work


def test_fit_cam_height_from_road_points_despite_outliers():
    """Path along +z at y = 0, up = -y (COLMAP), 2 scene units per metre; road 1.6 m below with a 2%
    camber, plus kerbs, a car roof and points far to the side that must not count."""
    rng = np.random.default_rng(0)
    P = np.stack([np.zeros(40), np.zeros(40), np.arange(40) * 0.5], 1)
    up, mpu = np.array([0, -1.0, 0]), 0.5
    lat = rng.uniform(-3.5, 3.5, 200)
    road = np.stack([lat, 1.6 + 0.02 * lat, rng.uniform(0, 19, 200)], 1) / mpu
    road[:, 2] *= mpu  # z already in scene units
    junk = np.stack([rng.uniform(-3, 3, 60), rng.uniform(0.9, 1.3, 60), rng.uniform(0, 19, 60)], 1)
    junk[:, :2] /= mpu
    far = np.stack([np.full(30, 20.0), np.full(30, 1.0), rng.uniform(0, 19, 30)], 1) / mpu
    h = recon.fit_cam_height(np.concatenate([road, junk, far]), P, up, mpu)
    assert h == pytest.approx(1.6, abs=0.02)
    assert recon.fit_cam_height(road[:20], P, up, mpu) is None  # too few to trust: use the default
    assert recon.fit_cam_height(np.zeros((0, 3)), P, up, mpu) is None


def test_export_path_and_nvs(tmp_path):
    work = make_work(tmp_path)
    path = json.loads(recon.export_path(work, work / "dense").read_text())
    assert path["camHeight"] == 1.4 and path["camHeightSource"] == "default"  # no SfM points here
    assert len(path["frames"]) == 6
    assert np.allclose(path["up"], [0, -1, 0])
    for f in path["frames"]:
        assert np.allclose(f["f"], [0, 0, 1], atol=1e-6)
    assert [f["t"] for f in path["frames"]] == [0, 1, 2, 3, 4, 5]
    # No telemetry: assume 11 m/s over the registered span (t 0..5).
    units = np.linalg.norm(np.diff([f["p"] for f in path["frames"]], axis=0), axis=1).sum()
    assert path["metresPerUnit"] == pytest.approx(11.0 * 5 / units, rel=1e-4)
    assert path["scaleSource"] == "assumed-speed"

    views = json.loads((recon.export_nvs(work, fps=2.0) / "views.json").read_text())
    refs = views["references"]
    assert len(refs) == 6 and refs[3]["camera"] == "front" and refs[3]["t"] == 3
    assert np.allclose(np.array(refs[3]["c2w"])[:3, 3], [0, 0, 3])
    assert refs[0]["intrinsics"] == {"w": 640, "h": 480, "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    frames = views["target"]["frames"]
    assert len(frames) == 10
    for fr in frames:
        c2w = np.array(fr["c2w"])
        R = c2w[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-5)
        assert R[2, 2] > 0.8  # chase cam looks forward (+z)
        assert R[:, 1] @ np.array(path["up"]) < 0  # OpenCV y points down


def test_export_path_scale_from_telemetry(tmp_path):
    tel = [{"t": i / 10, "mph": 22.3694} for i in range(50)]  # 10 m/s for 5 s = 50 m
    work = make_work(tmp_path, tel=tel)
    path = json.loads(recon.export_path(work, work / "dense").read_text())
    units = np.linalg.norm(np.diff([f["p"] for f in path["frames"]], axis=0), axis=1).sum()
    assert path["metresPerUnit"] == pytest.approx(50.0 / units, rel=1e-4)
    assert path["scaleSource"] == "telemetry"


def test_scale_fallbacks_are_explicit():
    assert recon.scale_estimate([], 0, 5, 0) == (1.0, "default-stationary")
    assert recon.scale_estimate([{"t": 0, "mph": 0}, {"t": 5, "mph": 0}], 0, 5, 2) == (
        1.0,
        "default-insufficient-telemetry",
    )
    for units in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError, match="scale requires"):
            recon.scale_estimate([], 0, 5, units)


def test_stationary_path_does_not_export_invalid_headings(tmp_path):
    work = make_work(tmp_path)
    model = work / "dense/sparse/images.txt"
    lines = model.read_text().splitlines()
    for i, line in enumerate(lines):
        if "front/" in line:
            fields = line.split()
            fields[5:8] = ["0", "0", "0"]
            lines[i] = " ".join(fields)
    model.write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="stationary"):
        recon.export_path(work, work / "dense")
    assert not (work / "viewer/path.json").exists()


def sei_rows(fps, seconds, mps=10.0, seq0=5000):
    return [{"frame_seq_no": seq0 + i, "vehicle_speed_mps": mps} for i in range(int(fps * seconds) + 1)]


@pytest.mark.parametrize("fps", [30, 36])
def test_export_path_scale_independent_of_video_fps(tmp_path, fps):
    """10 m/s; telemetry covers 0..6 s but only frames at 0..5 s are registered: 50 m, not 60 m."""
    tel = prep.telemetry(sei_rows(fps, 8), fps, start=0, dur=6)
    assert tel[-1]["t"] == pytest.approx(6, abs=0.05)
    work = make_work(tmp_path, tel=tel)
    path = json.loads(recon.export_path(work, work / "dense").read_text())
    units = np.linalg.norm(np.diff([f["p"] for f in path["frames"]], axis=0), axis=1).sum()
    # rel 0.5%: telemetry.json rounds mph to 0.1 (22.37 -> 22.4). The old "10 Hz" sum under-read by
    # ~8% at 36 fps (samples every 0.111 s) and ~18% at 30 fps (float error skipped some to 0.133 s).
    assert path["metresPerUnit"] == pytest.approx(50.0 / units, rel=5e-3)


def test_metres_driven_integrates_real_timestamps():
    # irregular sampling, speed ramps 0 -> 20 m/s over 10 s: distance = 100 m
    ts = [0, 0.3, 1.1, 2.0, 4.7, 5.0, 8.2, 10.0]
    tel = [{"t": t, "mph": 2.0 * t * 2.23694} for t in ts]
    assert recon.metres_driven(tel, 0, 10) == pytest.approx(100.0, rel=1e-6)
    assert recon.metres_driven(tel, 5, 10) == pytest.approx(75.0, rel=1e-6)  # restricted to the span
    assert recon.metres_driven(tel[:1], 0, 10) == 0.0


def write_images_bin(model, count, padding=0):
    model.mkdir(parents=True)
    (model / "images.bin").write_bytes(struct.pack("<Q", count) + b"\0" * padding)


def test_registered_reads_image_count_not_file_size(tmp_path):
    sparse = tmp_path / "work" / "sparse"
    write_images_bin(sparse / "0", 3, padding=10_000)  # bigger file, fewer images
    write_images_bin(sparse / "1", 5, padding=10)
    (sparse / "2").mkdir()  # failed model: no images.bin
    assert [recon._registered(sparse / d) for d in "012"] == [3, 5, 0]
    assert bench._sfm_model(tmp_path / "work").name == "1"


FAKE_BRUSH = """#!{python}
import os, sys
from pathlib import Path
a = sys.argv[1:]
val = lambda k: a[a.index(k) + 1]
steps, every, out = int(val("--total-steps")), int(val("--export-every")), Path(val("--export-path"))
mode = os.environ.get("FAKE_BRUSH_MODE", "ok")
if mode == "nothing":
    sys.exit(0)
last = steps if mode == "ok" else steps // 2
for it in range(every, last + 1, every):
    (out / val("--export-name").replace("{{iter}}", str(it))).write_text(f"ply {{it}}")
if "--eval-save-to-disk" in a and mode == "ok":
    (out / f"eval_{{steps}}").mkdir()
"""


@pytest.fixture
def fake_brush(tmp_path, monkeypatch):
    """A tiny Brush stand-in (via $BRUSH) plus a work dir holding a previous run's stale outputs."""
    exe = tmp_path / "bin" / "brush_app"
    exe.parent.mkdir()
    exe.write_text(FAKE_BRUSH.format(python=sys.executable))
    exe.chmod(0o755)
    monkeypatch.setenv("BRUSH", str(exe))
    work = tmp_path / "w"
    (work / "viewer").mkdir(parents=True)
    (work / "splat" / "eval_999").mkdir(parents=True)
    (work / "splat" / "splat_99999.ply").write_text("stale")
    return work


def test_train_fresh_splat_dir_and_final_export(fake_brush):
    work = fake_brush
    dest = recon.train(work, work / "dense", 8000, viewer=False, eval_every=8)
    assert dest.read_text() == "ply 8000"
    names = sorted(p.name for p in (work / "splat").iterdir())
    assert names == ["eval_8000", "splat_2000.ply", "splat_4000.ply", "splat_6000.ply", "splat_8000.ply"]


def test_export_interval_always_hits_final_step():
    for steps in (1000, 2500, 7000, 15000, 30000, 30001):
        assert steps % recon.export_every(steps) == 0


def test_train_fails_clearly_when_brush_exports_nothing(fake_brush, monkeypatch):
    monkeypatch.setenv("FAKE_BRUSH_MODE", "nothing")
    with pytest.raises(SystemExit, match="exported nothing"):
        recon.train(fake_brush, fake_brush / "dense", 8000, viewer=False)
    assert not (fake_brush / "splat" / "splat_99999.ply").exists()  # stale output is never picked up


def test_train_fails_clearly_when_final_splat_missing(fake_brush, monkeypatch):
    monkeypatch.setenv("FAKE_BRUSH_MODE", "early")
    with pytest.raises(SystemExit, match="splat_8000.ply"):
        recon.train(fake_brush, fake_brush / "dense", 8000, viewer=False)
    assert not (fake_brush / "viewer" / "scene.ply").exists()

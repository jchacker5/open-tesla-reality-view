import json

import numpy as np
import pytest

from tsplat import recon


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


def test_export_path_and_nvs(tmp_path):
    work = make_work(tmp_path)
    path = json.loads(recon.export_path(work, work / "dense").read_text())
    assert len(path["frames"]) == 6
    assert np.allclose(path["up"], [0, -1, 0])
    for f in path["frames"]:
        assert np.allclose(f["f"], [0, 0, 1], atol=1e-6)
    assert [f["t"] for f in path["frames"]] == [0, 1, 2, 3, 4, 5]
    # No telemetry: assume 11 m/s over the clip duration.
    units = np.linalg.norm(np.diff([f["p"] for f in path["frames"]], axis=0), axis=1).sum()
    assert path["metresPerUnit"] == pytest.approx(11.0 * 6 / units, rel=1e-4)

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

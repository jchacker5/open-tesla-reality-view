import json

import numpy as np
import pytest
from PIL import Image

from tsplat import densify

W, H, F = 80, 60, 40.0
CAM_H = 1.4  # ground plane y = +1.4 (COLMAP: y down), wall at z = WALL_Z
WALL_Z = 15.0


def true_depth(cz: float) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dy = (ys - H / 2) / F
    ground = np.where(dy > 1e-6, CAM_H / np.maximum(dy, 1e-6), np.inf)
    return np.minimum(ground, WALL_Z - cz)


def make_dense(tmp_path, n=5):
    """Front cameras at z = 0..n-1 looking down +z (identity rotation) at a road and a wall."""
    work = tmp_path / "w"
    dense = work / "dense"
    sp = dense / "sparse"
    (dense / "images" / "front").mkdir(parents=True)
    sp.mkdir(parents=True)
    (sp / "cameras.txt").write_text(f"# cams\n1 PINHOLE {W} {H} {F} {F} {W / 2} {H / 2}\n")
    rng = np.random.default_rng(0)
    pts = np.concatenate(
        [
            np.stack([rng.uniform(-3, 3, 60), np.full(60, CAM_H), rng.uniform(n + 2, 14, 60)], 1),
            np.stack([rng.uniform(-5, 5, 60), rng.uniform(-3, 1, 60), np.full(60, WALL_Z)], 1),
        ]
    )
    (sp / "points3D.txt").write_text(
        "# points\n" + "\n".join(f"{i + 1} {x} {y} {z} 128 128 128 0.5 1 0" for i, (x, y, z) in enumerate(pts)) + "\n"
    )
    lines = ["# images"]
    for i in range(n):
        name = f"front/front_{i:05d}.jpg"
        lines.append(f"{i + 1} 1 0 0 0 0 0 {-float(i)} 1 {name}")
        lines.append(" ".join(f"0 0 {j + 1}" for j in range(len(pts))))
        img = np.zeros((H, W, 3), np.uint8)
        img[..., 1] = 200
        img[0, 0, 0] = i  # lets the fake depth model know which view it is looking at
        Image.fromarray(img).save(dense / "images" / name, format="PNG")  # lossless, despite the name
    (sp / "images.txt").write_text("\n".join(lines) + "\n")
    frames = [{"t": i, "p": [0, 0, float(i)], "f": [0, 0, 1]} for i in range(n)]
    (work / "viewer").mkdir()
    (work / "viewer" / "path.json").write_text(json.dumps({"up": [0, -1, 0], "metresPerUnit": 1.0, "frames": frames}))
    return work, dense


def fake_depth(img):
    """Affine-ambiguous inverse depth (what Depth Anything returns): 3/z + 0.2."""
    i = int(np.asarray(img)[0, 0, 0])
    return 3.0 / true_depth(float(min(i, 4))) + 0.2


def test_fit_inverse_depth_recovers_scale_and_shift_despite_outliers():
    rng = np.random.default_rng(1)
    d = rng.uniform(0.1, 2.0, 200)
    inv_z = 0.5 * d + 0.05
    inv_z[:30] *= rng.uniform(2, 5, 30)  # 15% gross outliers (bad matches)
    s, b = densify.fit_inverse_depth(d, inv_z)
    assert s == pytest.approx(0.5, rel=0.02) and b == pytest.approx(0.05, abs=0.01)
    assert densify.fit_inverse_depth(d[:5], inv_z[:5]) is None


def test_voxel_downsample_averages_per_voxel_with_per_point_sizes():
    xyz = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.5, 0, 0], [0.51, 0, 0]])
    rgb = np.array([[0, 0, 0], [100, 100, 100], [10, 10, 10], [10, 10, 10]], np.uint8)
    out, col = densify.voxel_downsample(xyz, rgb, np.array([0.05, 0.05, 0.001, 0.001]))
    assert len(out) == 3  # the first two merge; the last two are 1 cm apart in 1 mm voxels
    merged = np.nonzero(np.isclose(out[:, 0], 0.015))[0]
    assert len(merged) == 1 and col[merged[0]].tolist() == [50, 50, 50]


def test_dense_init_backprojects_onto_the_scene_and_adds_a_ribbon(tmp_path):
    work, dense = make_dense(tmp_path)
    before = (dense / "sparse" / "points3D.txt").read_text().splitlines()
    counts = densify.run(work, dense, depth_fn=fake_depth)
    assert counts["depth"] > 100 and counts["ribbon"] > 100 and counts["sfm"] == 120
    lines = (dense / "sparse" / "points3D.txt").read_text().splitlines()
    assert lines[: len(before)] == before and all(line.strip() for line in lines)
    new = [line.split() for line in lines[len(before) :]]
    assert len(new) == counts["added"] == counts["depth"] + counts["ribbon"]
    ids = [int(p[0]) for p in new]
    assert ids[0] == 121 and len(set(ids)) == len(ids)
    assert all(len(p) == 8 for p in new)  # x y z r g b error, empty track
    xyz = np.array([[float(v) for v in p[1:4]] for p in new])
    on_ground = np.abs(xyz[:, 1] - CAM_H) < 0.05
    on_wall = np.abs(xyz[:, 2] - WALL_Z) < 0.15
    assert (on_ground | on_wall).mean() > 0.97
    ribbon = xyz[counts["depth"] :]
    assert np.allclose(ribbon[:, 1], CAM_H) and ribbon[:, 0].min() >= -6.01 and ribbon[:, 0].max() <= 6.01
    rgb = np.array([[int(v) for v in p[4:7]] for p in new[counts["depth"] :]])
    assert np.abs(rgb[:, 1] - 200).max() < 8  # coloured from the images


def test_dense_init_never_uses_held_out_views(tmp_path):
    """recon --eval's held-out (and guard) views neither get back-projected nor colour the ribbon."""
    work, dense = make_dense(tmp_path)
    for i in (0, 1):  # paint the held-out views red, so any colour taken from them shows
        img = np.zeros((H, W, 3), np.uint8)
        img[..., 0] = 255
        img[0, 0, 0] = i
        Image.fromarray(img).save(dense / "images" / f"front/front_{i:05d}.jpg", format="PNG")
    seen = []

    def depth(img):
        seen.append(int(np.asarray(img)[0, 0, 0]))
        return fake_depth(img)

    counts = densify.run(work, dense, depth_fn=depth, exclude={"front_00000", "front_00001"})
    assert sorted(seen) == [2, 3, 4] and counts["excluded_views"] == 2
    lines = (dense / "sparse" / "points3D.txt").read_text().splitlines()[-counts["added"] :]
    rgb = np.array([[int(v) for v in ln.split()[4:7]] for ln in lines])
    assert counts["ribbon"] > 0 and rgb[:, 0].max() < 50  # nothing red: no colour from views 0 and 1


def test_dense_init_skips_masked_and_sky_pixels(tmp_path):
    work, dense = make_dense(tmp_path)
    (dense / "images" / "masks").mkdir()
    (dense / "sky").mkdir()
    for i in range(5):
        keep = np.full((H, W), 255, np.uint8)
        keep[:, : W // 2] = 0  # left half masked (ego/movers)
        Image.fromarray(keep).save(dense / "images" / "masks" / f"front_{i:05d}.png")
        Image.fromarray(np.full((H, W), 255, np.uint8)).save(dense / "sky" / f"front_{i:05d}.png")  # all sky
    counts = densify.run(work, dense, depth_fn=fake_depth)
    assert counts["depth"] == 0 and counts["ribbon"] == 0

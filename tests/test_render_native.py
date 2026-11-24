"""The C++ splat renderer against an independent brute-force numpy implementation of the same forward
pass (no tiles, no sort keys, every pixel against every Gaussian), plus `otrv snapshot` end to end.

OTRV_BRUSH_EVAL_RUN=work/<run> (a run trained with `otrv recon --eval`) also renders every held-out
view and compares it with Brush's own render of that view.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from tsplat import snapshot

try:
    import otrv_native
except ImportError:
    if os.environ.get("OTRV_REQUIRE_NATIVE") == "1":
        raise
    pytest.skip("otrv_native not installed (uv sync --extra native)", allow_module_level=True)

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435]  # fmt: skip


def sh_basis(d, degree):
    x, y, z = d
    b = [C0]
    if degree > 0:
        b += [-C1 * y, C1 * z, -C1 * x]
    if degree > 1:
        xx, yy, zz = x * x, y * y, z * z
        b += [C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy)]
    if degree > 2:
        b += [
            C3[0] * y * (3 * xx - yy),
            C3[1] * x * y * z,
            C3[2] * y * (4 * zz - xx - yy),
            C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
            C3[4] * x * (4 * zz - xx - yy),
            C3[5] * z * (xx - yy),
            C3[6] * x * (xx - 3 * yy),
        ]
    return np.array(b)


def quat_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def reference_render(scene, cam, bg=(0.0, 0.0, 0.0)):
    """Brute force, float64: project, cull, sort by depth, composite every pixel against every splat."""
    pos, ls, rot, op, sh, deg = scene
    W, H = cam.width, cam.height
    R, t = np.array(cam.R), np.array(cam.t)
    centre = -R.T @ t
    k = (deg + 1) ** 2
    tiles_x, tiles_y = math.ceil(W / 16), math.ceil(H / 16)
    lim_x, lim_y = 1.3 * 0.5 * W / cam.fx, 1.3 * 0.5 * H / cam.fy
    proj = []
    for i in range(len(op)):
        x, y, z = R @ pos[i] + t
        if z < 0.01:
            continue
        T = R @ quat_matrix(rot[i]) @ np.diag(np.exp(ls[i]))
        tx, ty = np.clip(x / z, -lim_x, lim_x) * z, np.clip(y / z, -lim_y, lim_y) * z
        J = np.array([[cam.fx / z, 0, -cam.fx * tx / z**2], [0, cam.fy / z, -cam.fy * ty / z**2]])
        cov = J @ T @ T.T @ J.T + 0.3 * np.eye(2)
        det = np.linalg.det(cov)
        if det <= 0:
            continue
        mid = 0.5 * (cov[0, 0] + cov[1, 1])
        r = math.ceil(3 * math.sqrt(mid + math.sqrt(max(0.1, mid * mid - det))))
        mx, my = cam.fx * x / z + cam.cx, cam.fy * y / z + cam.cy
        x0, x1 = (min(max(math.floor(v), 0), tiles_x) for v in ((mx - r) / 16, (mx + r + 15) / 16))
        y0, y1 = (min(max(math.floor(v), 0), tiles_y) for v in ((my - r) / 16, (my + r + 15) / 16))
        if x0 >= x1 or y0 >= y1:
            continue
        d = pos[i] - centre
        rgb = np.maximum(0, 0.5 + sh_basis(d / np.linalg.norm(d), deg) @ sh[i].reshape(k, 3))
        proj.append((np.float32(z), i, mx, my, np.linalg.inv(cov), 1 / (1 + math.exp(-op[i])), rgb, (x0, y0, x1, y1)))
    proj.sort(key=lambda p: (p[0], p[1]))
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    tile_x, tile_y = (xx // 16).astype(int), (yy // 16).astype(int)
    C = np.zeros((H, W, 3))
    Tr = np.ones((H, W))
    live = np.ones((H, W), bool)
    for _, _, mx, my, conic, opacity, rgb, (x0, y0, x1, y1) in proj:
        dx, dy = mx - xx, my - yy
        power = -0.5 * (conic[0, 0] * dx * dx + conic[1, 1] * dy * dy) - conic[0, 1] * dx * dy
        a = np.minimum(0.99, opacity * np.exp(np.minimum(power, 0)))
        use = live & (power <= 0) & (a >= 1 / 255) & (tile_x >= x0) & (tile_x < x1) & (tile_y >= y0) & (tile_y < y1)
        nxt = Tr * (1 - a)
        stop = use & (nxt < 1e-4)
        live &= ~stop
        use &= ~stop
        C[use] += (a * Tr)[use, None] * rgb
        Tr = np.where(use, nxt, Tr)
    return C + Tr[..., None] * np.array(bg), 1 - Tr


def random_scene(rng, n, degree):
    k = (degree + 1) ** 2
    pos = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-2, 2, n), rng.uniform(1, 9, n)])
    pos[: n // 10, 2] = rng.uniform(-3, 0.005, n // 10)  # some behind or at the camera
    ls = np.log(rng.uniform(0.02, 0.6, (n, 3)))
    rot = rng.normal(size=(n, 4))
    op = rng.normal(0, 2, n)
    sh = rng.normal(0, 0.4, (n, k * 3))
    return pos, ls, rot, op, sh, degree


def native(scene):
    pos, ls, rot, op, sh, deg = scene
    return otrv_native.splats_from_arrays(
        pos.ravel().tolist(), ls.ravel().tolist(), rot.ravel().tolist(), op.tolist(), sh.ravel().tolist(), deg
    )


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_matches_brute_force(degree):
    rng = np.random.default_rng(degree)
    scene = random_scene(rng, 250, degree)
    cam = otrv_native.Camera.look_at((0.3, -0.4, -1.5), (0, 0, 5), (0, -1, 0), 70, 83, 61)  # partial edge tiles
    rgb, alpha, stats = otrv_native.render(native(scene), cam, background=(0.1, 0.2, 0.3), threads=4)
    want_rgb, want_alpha = reference_render(scene, cam, bg=(0.1, 0.2, 0.3))
    assert stats["visible"] > 100
    diff = np.abs(rgb - want_rgb)
    # float32 raster vs float64 reference: tiny everywhere, and no pixel may flip an early-termination decision
    assert np.percentile(diff, 99.9) < 1e-4, np.percentile(diff, 99.9)
    assert diff.max() < 2e-3, diff.max()
    assert np.abs(alpha - want_alpha).max() < 2e-3


def test_camera_conventions():
    cam = otrv_native.Camera.look_at((1, 2, 3), (1, 2, 10), (0, -1, 0), 90, 100, 50)
    assert np.allclose(cam.centre, (1, 2, 3))
    assert cam.fy == pytest.approx(25) and cam.cx == 50 and cam.cy == 25
    q = (0.9238795, 0.0, 0.3826834, 0.0)  # 45 degrees about y
    c = otrv_native.Camera.from_colmap(q, (0.0, 0.0, 2.0), 64, 48, 50, 50, 32, 24)
    assert np.allclose(np.array(c.R) @ np.array([1.0, 0, 0]), [math.sqrt(0.5), 0, -math.sqrt(0.5)], atol=1e-6)


def test_load_splats_rejects_garbage(tmp_path):
    p = tmp_path / "x.ply"
    p.write_text("not a ply")
    with pytest.raises(RuntimeError, match="not a PLY"):
        otrv_native.load_splats(str(p))


def write_scene(path: Path, scene):
    pos, ls, rot, op, sh, deg = scene
    k = (deg + 1) ** 2
    names = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    names += ["f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(3 * (k - 1))]
    shc = sh.reshape(-1, k, 3)
    rest = shc[:, 1:, :].transpose(0, 2, 1).reshape(len(op), -1)  # channel-major in the file
    cols = np.column_stack([pos, op, ls, rot, shc[:, 0, :], rest]).astype("<f4")
    head = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(op)}\n"
    head += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    path.write_bytes(head.encode() + cols.tobytes())


def test_snapshot_end_to_end(tmp_path, capsys):
    """A red ball where the chase camera aims: it must fill the middle of the frame."""
    mpu = 0.5  # metres per scene unit
    up = np.array([0.0, -1.0, 0.0])  # COLMAP-style: y down
    frames = [{"t": i / 6, "p": [0.0, 0.0, i * 0.5], "f": [0.0, 0.0, 1.0]} for i in range(30)]
    path = {"up": up.tolist(), "metresPerUnit": mpu, "camHeight": 1.4, "frames": frames}
    (tmp_path / "viewer").mkdir()
    (tmp_path / "viewer" / "path.json").write_text(json.dumps(path))
    p, f = snapshot.pose_at(path, 2.0)
    road = p - up * 1.4 / mpu
    aim = road + f * 4.5 / mpu + up * 0.6 / mpu
    red = (np.array([1.0, 0.0, 0.0]) - 0.5) / C0
    scene = (aim[None], np.log(np.full((1, 3), 0.5)), np.array([[1.0, 0, 0, 0]]), np.array([6.0]), red[None], 0)
    write_scene(tmp_path / "viewer" / "scene.ply", scene)
    out = snapshot.run(tmp_path, 2.0, "chase", None, 160, 90)
    from PIL import Image

    im = np.asarray(Image.open(out))
    assert im.shape == (90, 160, 3)
    assert im[45, 80, 0] > 200 and im[45, 80, 1] < 20  # red in the middle
    assert im[2, 2].max() < 5  # black background in the corner
    assert "snapshot:" in capsys.readouterr().out
    for view in ("driver", "drone"):
        assert snapshot.run(tmp_path, 1.0, view, tmp_path / f"{view}.png", 64, 36, quiet=True).exists()


@pytest.mark.skipif(not os.environ.get("OTRV_BRUSH_EVAL_RUN"), reason="set OTRV_BRUSH_EVAL_RUN=work/<run>")
def test_matches_brush_renders():
    """Every held-out view Brush rendered, re-rendered here from the same COLMAP camera."""
    from PIL import Image

    work = Path(os.environ["OTRV_BRUSH_EVAL_RUN"])
    evals = sorted(work.glob("splat/eval_*"))[-1]
    step = evals.name.split("_")[1]
    splats = otrv_native.load_splats(str(work / "splat" / f"splat_{step}.ply"))
    sparse = work / "dense" / "sparse"
    cams = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            v = line.split()
            cams[v[0]] = (int(v[2]), int(v[3]), *map(float, v[4:8]))
    poses = {}
    lines = [ln for ln in (sparse / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    for line in lines[::2]:
        v = line.split()
        poses[Path(v[9]).stem] = (tuple(map(float, v[1:5])), tuple(map(float, v[5:8])), v[8])
    psnrs = []
    for ref in sorted(evals.glob("*.png")):
        q, t, cid = poses[ref.stem]
        w, h, fx, fy, cx, cy = cams[cid]
        rgb, _, _ = otrv_native.render(splats, otrv_native.Camera.from_colmap(q, t, w, h, fx, fy, cx, cy))
        ours = np.clip(rgb, 0, 1) * 255 + 0.5
        theirs = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float64)
        mse = ((np.floor(ours) - theirs) ** 2).mean()
        psnrs.append(10 * math.log10(255**2 / mse) if mse else 99.0)
    assert psnrs and min(psnrs) > 60, psnrs

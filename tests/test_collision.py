"""Collision bake on a synthetic scene: flat ground 0.3 m above the path's road plane, plus a box wall."""

import json

import numpy as np
import pytest
from splatfix import make_ply, path_dict, write_path

from tsplat import __main__ as cli
from tsplat import collision, splats

GROUND_M = 0.3  # splat ground sits 0.3 m above where the path puts the road


def scene(mpu=1.0, wall_opacity=0.9):
    """Path along +z (up = -y, so right = +x). Ground splats every 0.2 m on x in [-12, 12] except a bare
    strip x < -8, z in [0, 20]; a 0.5 m thick wall at x = 5..5.5, z = 5..10, 0.3-2.8 m tall; a few
    high splats (tree canopy) at 5 m that must not count as wall."""
    u = 1 / mpu
    gx, gz = np.meshgrid(np.arange(-7.9, 12, 0.2), np.arange(0, 20, 0.2))
    ground = np.stack([gx.ravel(), np.full(gx.size, -GROUND_M), gz.ravel()], 1)
    wx, wy, wz = np.meshgrid(np.arange(5, 5.5, 0.1), np.arange(0.4, 2.8, 0.2), np.arange(5, 10, 0.1))
    wall = np.stack([wx.ravel(), -(wy.ravel() + GROUND_M), wz.ravel()], 1)
    canopy = np.stack([np.full(50, -4.0), np.full(50, -5.0), np.linspace(2, 4, 50)], 1)
    xyz = np.concatenate([ground, wall, canopy]) * u
    op = np.concatenate([np.full(len(ground), 0.9), np.full(len(wall), wall_opacity), np.full(len(canopy), 0.9)])
    return xyz, op, splats.EgoPath.from_dict(path_dict(n=21, step=1.0 * u, mpu=mpu))


def at(doc, key, x_m, z_m):
    """Value of grid `key` at (x, z) metres (collision.json's grid numbers are metres)."""
    if key == "height":
        cell, x0, z0, nx = doc["cell"], doc["x0"], doc["z0"], doc["nx"]
    else:
        cell, x0, z0, nx = doc["wallCell"], doc["wx0"], doc["wz0"], doc["wnx"]
    ix, iz = int((x_m - x0) // cell), int((z_m - z0) // cell)
    return doc[key][iz * nx + ix]


@pytest.mark.parametrize("mpu", [1.0, 0.05])
def test_ground_wall_and_basis(mpu):
    xyz, op, ego = scene(mpu)
    doc = collision.bake(xyz, op, ego)
    assert doc["units"] == "m" and doc["cell"] == 0.5 and doc["wallCell"] == 0.25 and doc["corridor"] == 12
    assert doc["metresPerUnit"] == mpu
    b = doc["basis"]
    assert np.allclose(b["origin"], [0, 0, 0], atol=1e-6)  # road under the first pose
    assert np.allclose(b["up"], [0, -1, 0]) and np.allclose(b["fwd"], [0, 0, 1]) and np.allclose(b["right"], [1, 0, 0])
    # right = fwd x up is the driver's right (here +x, the path runs along +z with up = -y), and the
    # (right, up, fwd) triple is left-handed, as documented
    r, u, f = (np.array(b[k]) for k in ("right", "up", "fwd"))
    assert np.allclose(np.cross(f, u), r) and np.allclose(np.cross(r, u), -f) and b["handedness"] == "left"
    assert len(doc["height"]) == doc["nx"] * doc["nz"] and len(doc["wallDist"]) == doc["wnx"] * doc["wnz"]

    # ground: splats where there are any, the path's road height (0) where there aren't
    assert at(doc, "height", 2.1, 10.1) == pytest.approx(GROUND_M, abs=0.01)
    assert at(doc, "height", -10.1, 10.1) == pytest.approx(0.0, abs=1e-6)
    # walls: inside the 0.5 m box is <= 0; 2 m from its face is ~2 m; the path is free; canopy isn't a wall
    assert -0.3 <= at(doc, "wallDist", 5.2, 7.6) <= 0
    assert at(doc, "wallDist", 3.1, 7.6) == pytest.approx(2.0, abs=0.3)
    assert at(doc, "wallDist", 0.1, 7.6) == pytest.approx(5.0, abs=0.3)
    assert at(doc, "wallDist", -4.0, 3.0) > 3
    # beyond the corridor (12 m from the path, here past its start) is wall, deeper the further out
    assert at(doc, "wallDist", -11.9, -11.9) < at(doc, "wallDist", -10.0, -10.0) < 0


def test_faint_splats_are_not_walls():
    xyz, op, ego = scene(wall_opacity=0.2)
    doc = collision.bake(xyz, op, ego)
    assert at(doc, "wallDist", 5.2, 7.6) > 4


def test_run_and_cli_write_viewer_collision_json(tmp_path, monkeypatch):
    xyz, op, _ = scene()
    work = tmp_path / "work" / "r"
    write_path(work / "viewer" / "path.json")
    (work / "meta.json").write_text("{}")
    splats.write_ply(work / "viewer" / "scene.ply", make_ply(xyz, opacity=op))
    monkeypatch.chdir(tmp_path)
    cli.main(["collision", "r"])
    doc = json.loads((work / "viewer" / "collision.json").read_text())
    assert {"cell", "x0", "z0", "nx", "nz", "height", "wallCell", "wx0", "wz0", "wnx", "wnz", "wallDist"} <= set(doc)
    assert set(doc["basis"]) == {"origin", "right", "up", "fwd", "handedness"} and doc["metresPerUnit"] == 1.0
    assert at(doc, "wallDist", 5.2, 7.6) <= 0


def test_cli_collision_needs_a_recon(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "work" / "r").mkdir(parents=True)
    (tmp_path / "work" / "r" / "meta.json").write_text("{}")
    with pytest.raises(SystemExit, match="otrv recon r"):
        cli.main(["collision", "r"])

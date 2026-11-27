import json
import struct

import numpy as np
import pytest

from tsplat import __main__ as cli
from tsplat import mesh, splats


def _make_scene_and_path(work):
    vd = work / "viewer"
    vd.mkdir(parents=True, exist_ok=True)
    # Path along +z at y = 0, up = [0, -1, 0], 1 unit = 1 metre -> road at y = 1.4.
    frames = [{"t": float(i), "p": [0.0, 0.0, float(i)], "f": [0.0, 0.0, 1.0]} for i in range(6)]
    (vd / "path.json").write_text(
        json.dumps({"up": [0.0, -1.0, 0.0], "metresPerUnit": 1.0, "camHeight": 1.4, "frames": frames})
    )

    pts, cols = [], []
    # Road splats across x in [-2, 2], z in [0, 5] at y = 1.4
    for x in np.linspace(-2.0, 2.0, 17):
        for z in np.linspace(0.0, 5.0, 21):
            pts.append([x, 1.4, z])
            cols.append([0.2, 0.2, 0.25])
    # Wall / structure splats at x = 3.0, z = 2.0, y in [0.0, 0.6] (above ground)
    for _ in range(4):
        pts.append([3.0, 0.4, 2.0])
        cols.append([0.8, 0.1, 0.1])
    # Ego-tube floater directly in the car's swept volume (must be excluded)
    for _ in range(4):
        pts.append([0.0, 0.8, 2.5])
        cols.append([0.0, 1.0, 0.0])

    n = len(pts)
    dt = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("opacity", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
        ]
    )
    arr = np.empty(n, dtype=dt)
    xyz = np.asarray(pts, np.float32)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["opacity"] = 2.0  # sigmoid(2) ~ 0.88
    c0 = 0.28209479177387814
    dc = (np.asarray(cols, np.float32) - 0.5) / c0
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = dc[:, 0], dc[:, 1], dc[:, 2]
    splats.write_ply(vd / "scene.ply", splats.Ply(arr, []))
    (work / "meta.json").write_text(json.dumps({"clip": "c", "fps": 1.0, "dur": 5.0, "cams": ["front"]}))
    return work


def test_export_mesh_ply_and_glb_and_excludes_ego_tube(tmp_path):
    work = _make_scene_and_path(tmp_path / "w")
    ply_out = mesh.run(work, fmt="ply")
    assert ply_out == work / "viewer" / "mesh.ply" and ply_out.is_file()
    raw = ply_out.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii")
    assert "element vertex " in header and "element face " in header
    assert "property uchar red" in header and "property list uchar int vertex_indices" in header

    # Ego-tube floater colour (green ~ [0, 1, 0]) must not appear in structure vertices.
    ply_obj = splats.read_ply(work / "viewer" / "scene.ply")
    ego = splats.EgoPath.load(work / "viewer" / "path.json")
    verts, cols, faces = mesh.build_mesh(ply_obj, ego)
    assert len(verts) > 0 and len(faces) > 0
    assert (faces >= 0).all() and (faces < len(verts)).all()
    assert not np.any((cols[:, 1] > 0.8) & (cols[:, 0] < 0.2))

    # Ground triangles must face upward (+ego.up).
    gverts, _, gfaces = mesh._extract_ground_mesh(
        ply_obj.xyz(), ply_obj.rgb(), ply_obj.opacity(), ego, cell_m=0.25, corridor_m=12.0
    )
    assert len(gfaces) > 0
    tri = gverts[gfaces]
    norms = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert np.all(norms @ ego.up > 0)

    glb_out = mesh.run(work, fmt="glb")
    assert glb_out == work / "viewer" / "mesh.glb" and glb_out.is_file()
    glb = glb_out.read_bytes()
    magic, version, total_len = struct.unpack("<III", glb[:12])
    assert magic == 0x46546C67 and version == 2 and total_len == len(glb)
    jlen, jtype = struct.unpack("<II", glb[12:20])
    assert jtype == 0x4E4F534A
    doc = json.loads(glb[20 : 20 + jlen].decode("utf-8"))
    assert doc["asset"]["version"] == "2.0"
    assert doc["meshes"][0]["primitives"][0]["attributes"] == {"POSITION": 0, "COLOR_0": 1}


def test_export_mesh_cli_and_visloc_passthrough(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    work = _make_scene_and_path(tmp_path / "work" / "drive1")
    out_glb = tmp_path / "custom.glb"
    cli.main(["export-mesh", "drive1", "--format", "glb", "--metres", "--out", str(out_glb)])
    assert out_glb.is_file() and out_glb.read_bytes()[:4] == b"glTF"

    (work / "visloc").mkdir()
    (work / "visloc" / "mesh.ply").write_bytes(b"visloc-ply")
    cli.main(["export-mesh", "drive1"])
    assert (work / "viewer" / "mesh.ply").read_bytes() == b"visloc-ply"


def test_export_mesh_rejects_empty_corridor(tmp_path):
    work = _make_scene_and_path(tmp_path / "w")
    # Overwrite scene with points far outside the corridor
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("opacity", "<f4")])
    arr = np.zeros(1, dtype=dt)
    arr["x"] = 500.0
    splats.write_ply(work / "viewer" / "scene.ply", splats.Ply(arr, []))
    with pytest.raises(SystemExit, match="no corridor splats"):
        mesh.run(work)


def test_structure_mesh_uses_metre_thresholds_when_mpu_scaled(tmp_path):
    # 1 scene unit = 0.1 m (10 units per metre): road is at y = 14.0 (1.4 m below origin).
    frames = [{"t": float(i), "p": [0.0, 0.0, float(i) * 10.0], "f": [0.0, 0.0, 1.0]} for i in range(6)]
    ego = splats.EgoPath.from_dict({"up": [0.0, -1.0, 0.0], "metresPerUnit": 0.1, "camHeight": 1.4, "frames": frames})
    # Place one cluster at lateral=3.0 m (x=30), height=1.5 m above road (y=-1.0), which MUST pass;
    # and one at lateral=15.0 m (x=150) outside 12 m corridor, which MUST be excluded.
    xyz = np.array(
        [[30.0, -1.0, 20.0], [30.1, -1.0, 20.0], [150.0, -1.0, 20.0], [150.1, -1.0, 20.0]],
        np.float32,
    )
    rgb = np.array([[0.9, 0.1, 0.1], [0.9, 0.1, 0.1], [0.1, 0.1, 0.9], [0.1, 0.1, 0.9]], np.float32)
    op = np.full(4, 0.9, np.float32)
    sverts, scols, sfaces = mesh._extract_structure_mesh(xyz, rgb, op, ego, voxel_m=0.35, corridor_m=12.0)
    assert len(sverts) > 0 and len(sfaces) > 0
    assert sverts[:, 0].max() < 50.0
    assert scols[:, 0].mean() > 0.7

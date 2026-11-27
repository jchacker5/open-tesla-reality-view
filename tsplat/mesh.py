"""Export a coloured triangle mesh (.ply or .glb) from a reconstructed Gaussian splat.

`otrv export-mesh <name>` builds a surface mesh of the road and surrounding structures inside
the reconstructed corridor from `viewer/scene.ply` + `viewer/path.json` (or converts
`visloc/mesh.ply` when `--backend visloc` produced one). Ego-tube splats are excluded so the
car's own volume does not appear as floating geometry.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import collision, splats

_CUBE_FACES = (
    # (neighbor_offset, four corner offsets in (dx, dy, dz) in CCW order seen from outside)
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0))),
)


def _cell_mean_rgb(cell_ids: np.ndarray, rgb: np.ndarray, n_cells: int, default: np.ndarray) -> np.ndarray:
    out = np.broadcast_to(default, (n_cells, 3)).copy()
    if len(cell_ids) == 0:
        return out
    counts = np.bincount(cell_ids, minlength=n_cells)
    valid = counts > 0
    for c in range(3):
        sums = np.bincount(cell_ids, weights=rgb[:, c], minlength=n_cells)
        out[valid, c] = sums[valid] / counts[valid]
    return out


def _extract_ground_mesh(
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    ego: splats.EgoPath,
    cell_m: float,
    corridor_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulated ground heightfield over the corridor in scene units."""
    u = 1.0 / ego.mpu
    basis = collision.frame(ego)
    road = ego.dense_road(step_m=max(0.15, cell_m))
    rx, ry, rz = collision.to_frame(road, basis)
    road_tree = cKDTree(np.stack([rx, rz], 1))
    corridor = corridor_m * u
    lo = np.array([rx.min(), rz.min()]) - corridor
    hi = np.array([rx.max(), rz.max()]) + corridor
    g = collision.Grid.covering(lo, hi, cell_m * u)

    sx, sy, sz = collision.to_frame(xyz, basis)
    lat, near = road_tree.query(np.stack([sx, sz], 1), workers=-1) if len(sx) else (np.zeros(0), np.zeros(0, int))
    cand = (lat <= corridor) & (opacity >= 0.15) & (np.abs(sy - ry[near]) < 1.5 * u)
    if not cand.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    gix, giz = g.index(sx[cand], sz[cand])
    cell_ids = giz * g.nx + gix
    n_cells = g.nx * g.nz
    height = collision._percentile_per_cell(cell_ids, sy[cand], n_cells, 15.0, min_count=2)
    cell_rgb = _cell_mean_rgb(cell_ids, rgb[cand], n_cells, np.array([0.25, 0.25, 0.27]))

    gcx, gcz = g.centres()
    glat, gnear = road_tree.query(np.stack([gcx.ravel(), gcz.ravel()], 1), workers=-1)
    in_corridor = glat.reshape(g.nz, g.nx) <= corridor
    height_2d = height.reshape(g.nz, g.nx)
    valid = ~np.isnan(height_2d) & in_corridor
    # Fill tiny 1-cell road gaps from the local road height when inside 4 m of the driven path.
    near_road = (glat.reshape(g.nz, g.nx) <= 4.0 * u) & ~valid
    height_2d = np.where(near_road, ry[gnear].reshape(g.nz, g.nx), height_2d)
    valid = valid | near_road

    if not valid.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)

    gx_flat = gcx.ravel()
    gz_flat = gcz.ravel()
    gy_flat = height_2d.ravel()
    pts_scene = (
        basis["origin"]
        + gx_flat[:, None] * basis["right"]
        + gy_flat[:, None] * basis["up"]
        + gz_flat[:, None] * basis["fwd"]
    )
    vid = np.full(n_cells, -1, np.int32)
    keep_idx = np.flatnonzero(valid.ravel())
    vid[keep_idx] = np.arange(len(keep_idx), dtype=np.int32)
    vid_2d = vid.reshape(g.nz, g.nx)

    v00 = vid_2d[:-1, :-1]
    v10 = vid_2d[:-1, 1:]
    v01 = vid_2d[1:, :-1]
    v11 = vid_2d[1:, 1:]
    quad_ok = (v00 >= 0) & (v10 >= 0) & (v01 >= 0) & (v11 >= 0)
    if not quad_ok.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)

    a, b, c, d = v00[quad_ok], v10[quad_ok], v11[quad_ok], v01[quad_ok]
    if np.dot(np.cross(basis["right"], basis["fwd"]), basis["up"]) >= 0:
        faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)], axis=0).astype(np.int32)
    else:
        faces = np.concatenate([np.stack([a, c, b], 1), np.stack([a, d, c], 1)], axis=0).astype(np.int32)
    return pts_scene[keep_idx].astype(np.float32), cell_rgb[keep_idx].astype(np.float32), faces


def _extract_structure_mesh(
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    ego: splats.EgoPath,
    voxel_m: float,
    corridor_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exposed-surface voxel mesh for opaque above-ground structures inside the corridor."""
    if len(xyz) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    u = 1.0 / ego.mpu
    _, height, _ = ego.nearest_road(xyz, across=True)
    lat, _, _ = ego.nearest_road(xyz, across=False)
    keep = (lat <= corridor_m) & (opacity >= 0.25) & (height >= 0.35) & (height <= 10.0)
    if not keep.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)

    pts = xyz[keep]
    cols = rgb[keep]
    step = voxel_m * u
    origin = pts.min(axis=0)
    ijk = np.floor((pts - origin) / step).astype(np.int32)

    occ_counts: dict[tuple[int, int, int], int] = {}
    occ_cols: dict[tuple[int, int, int], np.ndarray] = {}
    for cell_tuple, col in zip(map(tuple, ijk), cols, strict=False):
        occ_counts[cell_tuple] = occ_counts.get(cell_tuple, 0) + 1
        if cell_tuple in occ_cols:
            occ_cols[cell_tuple] += col
        else:
            occ_cols[cell_tuple] = col.astype(np.float64).copy()

    occupied = {k: occ_cols[k] / n for k, n in occ_counts.items() if n >= 2}
    if not occupied:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)

    verts: list[np.ndarray] = []
    vcols: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    for (ix, iy, iz), col in occupied.items():
        base_pt = origin + np.array([ix, iy, iz], np.float64) * step
        for (ox, oy, oz), corners in _CUBE_FACES:
            if (ix + ox, iy + oy, iz + oz) in occupied:
                continue
            v0 = len(verts)
            for cx, cy, cz in corners:
                verts.append(base_pt + np.array([cx, cy, cz], np.float64) * step)
                vcols.append(col)
            faces.append((v0, v0 + 1, v0 + 2))
            faces.append((v0, v0 + 2, v0 + 3))
    if not faces:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    return np.asarray(verts, np.float32), np.asarray(vcols, np.float32), np.asarray(faces, np.int32)


def build_mesh(
    ply: splats.Ply,
    ego: splats.EgoPath,
    cell_m: float = 0.25,
    voxel_m: float = 0.35,
    corridor_m: float = 12.0,
    metres: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(vertices [N,3] float32, colors [N,3] float32 in [0,1], faces [M,3] int32)`."""
    if cell_m <= 0 or voxel_m <= 0 or corridor_m <= 0:
        raise ValueError("cell_m, voxel_m, and corridor_m must be positive")
    xyz = ply.xyz()
    opacity = ply.opacity()
    rgb = ply.rgb()
    # Respect ego-path cleanup: exclude splats inside the car's swept tube.
    keep = ~ego.tube_mask(xyz)
    xyz, opacity, rgb = xyz[keep], opacity[keep], rgb[keep]

    gv, gc, gf = _extract_ground_mesh(xyz, rgb, opacity, ego, cell_m, corridor_m)
    sv, sc, sf = _extract_structure_mesh(xyz, rgb, opacity, ego, voxel_m, corridor_m)
    if len(gf) and len(sf):
        verts = np.concatenate([gv, sv], axis=0)
        cols = np.concatenate([gc, sc], axis=0)
        faces = np.concatenate([gf, sf + len(gv)], axis=0)
    elif len(gf):
        verts, cols, faces = gv, gc, gf
    else:
        verts, cols, faces = sv, sc, sf

    if len(verts) and metres:
        verts = ((verts.astype(np.float64) - ego.road[0]) * ego.mpu).astype(np.float32)
    return verts, np.clip(cols, 0.0, 1.0).astype(np.float32), faces


def write_mesh_ply(path: Path, verts: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary little-endian coloured triangle PLY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    v_arr = np.empty(
        len(verts),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    if len(verts):
        v_arr["x"], v_arr["y"], v_arr["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
        u8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
        v_arr["red"], v_arr["green"], v_arr["blue"] = u8[:, 0], u8[:, 1], u8[:, 2]

    f_arr = np.empty(len(faces), dtype=[("n", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4")])
    if len(faces):
        f_arr["n"] = 3
        f_arr["i0"], f_arr["i1"], f_arr["i2"] = faces[:, 0], faces[:, 1], faces[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(v_arr.tobytes())
        f.write(f_arr.tobytes())


def write_mesh_glb(path: Path, verts: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary glTF 2.0 (.glb) mesh with POSITION and COLOR_0 vertex attributes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = np.ascontiguousarray(verts, dtype="<f4")
    col = np.ascontiguousarray(colors, dtype="<f4")
    idx = np.ascontiguousarray(faces.ravel(), dtype="<u4")

    pos_bytes = pos.tobytes()
    col_bytes = col.tobytes()
    idx_bytes = idx.tobytes()
    bin_raw = pos_bytes + col_bytes + idx_bytes
    bin_pad = (-len(bin_raw)) % 4
    bin_chunk = bin_raw + b"\x00" * bin_pad

    vmin = pos.min(axis=0).tolist() if len(pos) else [0.0, 0.0, 0.0]
    vmax = pos.max(axis=0).tolist() if len(pos) else [0.0, 0.0, 0.0]

    gltf = {
        "asset": {"version": "2.0", "generator": "open-tesla-reality-view"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "indices": 2,
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": len(bin_raw)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bytes), "byteLength": len(col_bytes), "target": 34962},
            {
                "buffer": 0,
                "byteOffset": len(pos_bytes) + len(col_bytes),
                "byteLength": len(idx_bytes),
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": int(len(pos)),
                "type": "VEC3",
                "min": [float(x) for x in vmin],
                "max": [float(x) for x in vmax],
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": int(len(col)),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5125,
                "count": int(len(idx)),
                "type": "SCALAR",
            },
        ],
    }
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = (-len(json_bytes)) % 4
    json_chunk = json_bytes + b" " * json_pad

    total_len = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total_len))
        f.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        f.write(json_chunk)
        f.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        f.write(bin_chunk)


def run(
    work: Path,
    out: Path | None = None,
    fmt: str = "ply",
    cell_m: float = 0.25,
    voxel_m: float = 0.35,
    corridor_m: float = 12.0,
    metres: bool = False,
    source: str = "auto",
) -> Path:
    """Export `viewer/mesh.ply` or `.glb` for `work`."""
    fmt = fmt.lower()
    if fmt not in ("ply", "glb"):
        raise ValueError(f"format must be 'ply' or 'glb', got {fmt!r}")
    vd = work / "viewer"
    dest = out or (vd / f"mesh.{fmt}")

    visloc_mesh = work / "visloc" / "mesh.ply"
    if source in ("auto", "visloc") and visloc_mesh.is_file() and fmt == "ply":
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(visloc_mesh, dest)
        print(f"export-mesh: copied visloc mesh -> {dest}")
        return dest
    if source == "visloc" and not visloc_mesh.is_file():
        raise SystemExit(f"no visloc mesh at {visloc_mesh}; run `otrv recon --backend visloc` or omit --source visloc")

    ply = splats.read_ply(vd / "scene.ply")
    ego = splats.EgoPath.load(vd / "path.json")
    verts, cols, faces = build_mesh(ply, ego, cell_m=cell_m, voxel_m=voxel_m, corridor_m=corridor_m, metres=metres)
    if len(faces) == 0:
        raise SystemExit("export-mesh: no corridor splats left to mesh")
    if fmt == "glb":
        write_mesh_glb(dest, verts, cols, faces)
    else:
        write_mesh_ply(dest, verts, cols, faces)
    print(f"export-mesh: {len(verts)} vertices, {len(faces)} triangles -> {dest}")
    return dest

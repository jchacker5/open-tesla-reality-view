"""Optional `visloc-rs` (`gsplat_photos`) backend for Linux/Windows/macOS without COLMAP.

`otrv setup-visloc` downloads a SHA-256-verified release of `visloc-rs` (`gsplat_photos`), and
`otrv recon <name> --backend visloc` runs SfM, Gaussian-splat training, and mesh extraction in
one binary, then normalises its COLMAP sparse model into `work/<name>/dense/sparse/` without
requiring `colmap` on PATH.
"""

from __future__ import annotations

import os
import platform
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from . import brush

VERSION = "0.5.0"
BASE_URL = f"https://github.com/rsasaki0109/visloc-rs/releases/download/v{VERSION}/"
INSTALL_DIR = Path.home() / ".local" / "opt" / "visloc"

PINNED_SHA256 = {
    "visloc-rs-v0.5.0-aarch64-apple-darwin.tar.gz": (
        "ac16cd6c34d5d40dc98767be632855b184d1377bde7f8f34e2c4f2db6a79ae6c"
    ),
    "visloc-rs-v0.5.0-x86_64-unknown-linux-gnu.tar.gz": (
        "f6a9860221f2c23f1799c03528a012474a9e291122638917eaa866be09bad069"
    ),
    "visloc-rs-v0.5.0-x86_64-pc-windows-msvc.zip": ("a67a776f7a419234a2e69efd21f993131006c514253a02aec8538121f3b70a21"),
}

_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def asset_name(triple: str) -> str:
    ext = "zip" if "windows" in triple else "tar.gz"
    return f"visloc-rs-v{VERSION}-{triple}.{ext}"


def exe_name(triple: str) -> str:
    return "gsplat_photos.exe" if "windows" in triple else "gsplat_photos"


def default_path(install_dir: Path = INSTALL_DIR) -> Path:
    triple = brush.target() or "aarch64-apple-darwin"
    return install_dir / f"visloc-rs-v{VERSION}-{triple}" / exe_name(triple)


def path() -> str:
    return os.environ.get("VISLOC") or str(default_path())


def setup(
    install_dir: Path = INSTALL_DIR,
    triple: str | None = None,
    fetch=brush._download,
    pinned: dict[str, str] = PINNED_SHA256,
) -> Path:
    """Download the visloc-rs release archive + .sha256, verify both, and unpack to `install_dir`."""
    triple = triple or brush.target()
    if triple is None:
        raise SystemExit(
            f"visloc-rs v{VERSION} has no prebuilt binary for {platform.system()} {platform.machine()}. "
            "Build from https://github.com/rsasaki0109/visloc-rs and set VISLOC=/path/to/gsplat_photos."
        )
    name = asset_name(triple)
    with tempfile.TemporaryDirectory() as tmp:
        archive, sumfile = Path(tmp) / name, Path(tmp) / f"{name}.sha256"
        fetch(BASE_URL + name + ".sha256", sumfile)
        fetch(BASE_URL + name, archive)
        brush.check_digest(archive, name, brush.parse_sha256(sumfile.read_text(), name), pinned)
        print(f"sha256 ok (release .sha256{' + pinned' if name in pinned else ''}): {name}")
        brush.extract(archive, install_dir)
    exe = install_dir / f"visloc-rs-v{VERSION}-{triple}" / exe_name(triple)
    if not exe.exists():
        found = next(install_dir.rglob(exe_name(triple)), None)
        if found is None:
            raise SystemExit(f"extracted {name} but found no {exe_name(triple)} under {install_dir}")
        exe = found
    exe.chmod(exe.stat().st_mode | 0o111)
    return exe


def _read_cstr(fp) -> str:
    buf = bytearray()
    while (b := fp.read(1)) and b != b"\x00":
        buf.extend(b)
    return buf.decode("utf-8", "replace")


def convert_colmap_bin_to_txt(src_dir: Path, dst_dir: Path) -> None:
    """Pure-Python converter for COLMAP `cameras.bin`, `images.bin`, `points3D.bin` -> `.txt`."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if (src_dir / "cameras.txt").is_file() and (src_dir / "images.txt").is_file():
        for fn in ("cameras.txt", "images.txt", "points3D.txt"):
            if (src_dir / fn).is_file():
                shutil.copy2(src_dir / fn, dst_dir / fn)
        return

    with open(src_dir / "cameras.bin", "rb") as fp:
        n_cams = struct.unpack("<Q", fp.read(8))[0]
        cam_lines = ["# Camera list with one line of data per camera:"]
        for _ in range(n_cams):
            cam_id, model_id, w, h = struct.unpack("<IiQQ", fp.read(24))
            model_name, n_params = _CAMERA_MODELS.get(model_id, ("PINHOLE", 4))
            params = struct.unpack(f"<{n_params}d", fp.read(8 * n_params))
            cam_lines.append(f"{cam_id} {model_name} {w} {h} " + " ".join(f"{v:.8g}" for v in params))
        (dst_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n")

    with open(src_dir / "images.bin", "rb") as fp:
        n_imgs = struct.unpack("<Q", fp.read(8))[0]
        img_lines = ["# Image list with two lines of data per image:"]
        for _ in range(n_imgs):
            img_id = struct.unpack("<I", fp.read(4))[0]
            qvec = struct.unpack("<4d", fp.read(32))
            tvec = struct.unpack("<3d", fp.read(24))
            cam_id = struct.unpack("<I", fp.read(4))[0]
            name = _read_cstr(fp)
            n_pts2d = struct.unpack("<Q", fp.read(8))[0]
            pts2d = fp.read(24 * n_pts2d)
            pts_toks = []
            for k in range(n_pts2d):
                x, y, pid = struct.unpack_from("<ddq", pts2d, k * 24)
                pts_toks.append(f"{x:.6g} {y:.6g} {pid}")
            vals = " ".join(f"{v:.8g}" for v in (*qvec, *tvec))
            img_lines.append(f"{img_id} {vals} {cam_id} {name}")
            img_lines.append(" ".join(pts_toks))
        (dst_dir / "images.txt").write_text("\n".join(img_lines) + "\n")

    pts_bin = src_dir / "points3D.bin"
    if pts_bin.is_file():
        with open(pts_bin, "rb") as fp:
            n_pts = struct.unpack("<Q", fp.read(8))[0]
            pt_lines = ["# 3D point list with one line of data per point:"]
            for _ in range(n_pts):
                pid, x, y, z, r, g, b, err = struct.unpack("<QdddBBBd", fp.read(43))
                track_len = struct.unpack("<Q", fp.read(8))[0]
                fp.read(8 * track_len)
                pt_lines.append(f"{pid} {x:.8g} {y:.8g} {z:.8g} {r} {g} {b} {err:.6g}")
            (dst_dir / "points3D.txt").write_text("\n".join(pt_lines) + "\n")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_inputs(work: Path) -> tuple[Path, Path]:
    """Flatten `<cam>/<idx>.jpg` into `visloc_in/images/<cam>_<idx>.jpg` (+ matching masks)."""
    in_root = work / "visloc_in"
    shutil.rmtree(in_root, ignore_errors=True)
    img_dir = in_root / "images"
    mask_dir = in_root / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for src in sorted((work / "images").glob("*/*.jpg")):
        cam = src.parent.name
        flat = f"{cam}_{src.name}"
        _link_or_copy(src, img_dir / flat)
        m = work / "masks" / cam / f"{src.name}.png"
        if m.is_file():
            _link_or_copy(m, mask_dir / f"{flat}.png")
    return img_dir, mask_dir


def _rewrite_image_names(images_txt: Path, cams: list[str]) -> None:
    """Map visloc's flat `<cam>_<idx>.jpg` back to `<cam>/<cam>_<idx>.jpg` expected by export_path/bench."""
    ordered = sorted(cams, key=len, reverse=True)
    lines = images_txt.read_text().splitlines()
    out, n = [], 0
    for line in lines:
        if not line.startswith("#"):
            n += 1
            if n % 2 == 1:
                head, name = line.rsplit(" ", 1)
                stem = Path(name).name
                cam = next((c for c in ordered if stem.startswith(f"{c}_")), None)
                if cam:
                    line = f"{head} {cam}/{stem}"
        out.append(line)
    images_txt.write_text("\n".join(out) + "\n")


def run(work: Path, steps: int = 15000, rules=None) -> Path:
    """Run `gsplat_photos`, normalise its sparse model into `dense/sparse/`, and export `viewer/scene.ply`."""
    exe = path()
    if not Path(exe).is_file():
        raise SystemExit(f"visloc-rs gsplat_photos not found at {exe}; run `otrv setup-visloc` or set VISLOC=...")
    img_dir, mask_dir = stage_inputs(work)
    v_out = work / "visloc"
    shutil.rmtree(v_out, ignore_errors=True)
    v_out.mkdir(parents=True)
    cmd = [
        exe,
        "--images",
        str(img_dir),
        "--masks",
        str(mask_dir),
        "--out",
        str(v_out),
        "--steps",
        str(steps),
    ]
    print("$", " ".join(cmd[:4]), "...")
    subprocess.run(cmd, check=True)

    sparse0 = v_out / "sparse" / "0"
    if not sparse0.is_dir():
        raise SystemExit(f"visloc-rs produced no sparse model in {sparse0}")
    dense = work / "dense"
    shutil.rmtree(dense, ignore_errors=True)
    convert_colmap_bin_to_txt(sparse0, dense / "sparse")

    import json

    cams = json.loads((work / "meta.json").read_text()).get("cams", []) if (work / "meta.json").is_file() else []
    _rewrite_image_names(dense / "sparse" / "images.txt", cams)
    for src in sorted(img_dir.glob("*.jpg")):
        cam = next((c for c in sorted(cams, key=len, reverse=True) if src.name.startswith(f"{c}_")), None)
        if cam:
            _link_or_copy(src, dense / "images" / cam / src.name)

    splat_src = v_out / "scene.ply"
    if not splat_src.is_file():
        raise SystemExit(f"visloc-rs produced no scene.ply at {splat_src}")
    dest = work / "viewer" / "scene.ply"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for stale in (work / "clean.json", work / "viewer" / "scene_raw.ply", dest):
        stale.unlink(missing_ok=True)
    if rules and (rules.prune or rules.dehaze) and (work / "viewer" / "path.json").exists():
        from . import clean

        clean.run(work, splat_src, rules)
    else:
        shutil.copy2(splat_src, dest)
    return dense

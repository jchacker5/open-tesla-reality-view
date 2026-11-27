"""Tests for the cross-platform visloc-rs (gsplat_photos) backend."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import struct
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from tsplat import __main__ as cli
from tsplat import doctor, splats, visloc


def _write_colmap_bin(sparse_dir: Path, image_names: list[str]) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    # cameras.bin: 1 PINHOLE camera (model_id=1, w=1280, h=960, params=fx,fy,cx,cy)
    with open(sparse_dir / "cameras.bin", "wb") as fp:
        fp.write(struct.pack("<Q", 1))
        fp.write(struct.pack("<IiQQ", 1, 1, 1280, 960))
        fp.write(struct.pack("<4d", 900.0, 900.0, 640.0, 480.0))

    # images.bin: one camera entry per image name along +Z
    with open(sparse_dir / "images.bin", "wb") as fp:
        fp.write(struct.pack("<Q", len(image_names)))
        for idx, name in enumerate(image_names, start=1):
            fp.write(struct.pack("<I", idx))
            fp.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))
            fp.write(struct.pack("<3d", 0.0, 0.0, float(-idx * 2)))
            fp.write(struct.pack("<I", 1))
            fp.write(name.encode("utf-8") + b"\x00")
            # 1 2D point
            fp.write(struct.pack("<Q", 1))
            fp.write(struct.pack("<ddq", 10.5, 20.5, idx))

    # points3D.bin: 1 point
    with open(sparse_dir / "points3D.bin", "wb") as fp:
        fp.write(struct.pack("<Q", 1))
        fp.write(struct.pack("<QdddBBBd", 1, 0.0, 0.0, 2.0, 128, 128, 128, 0.5))
        fp.write(struct.pack("<Q", 1))
        fp.write(struct.pack("<II", 1, 0))


def _write_minimal_splat_ply(path: Path, n: int = 8) -> None:
    xyz = np.column_stack([np.linspace(-2.0, 2.0, n), np.zeros(n), np.linspace(0.0, 10.0, n)]).astype(np.float32)
    fields = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"),
        ("scale_1", "<f4"),
        ("scale_2", "<f4"),
        ("f_dc_0", "<f4"),
        ("f_dc_1", "<f4"),
        ("f_dc_2", "<f4"),
        ("rot_0", "<f4"),
        ("rot_1", "<f4"),
        ("rot_2", "<f4"),
        ("rot_3", "<f4"),
    ]
    arr = np.zeros(n, dtype=np.dtype(fields))
    arr["x"] = xyz[:, 0]
    arr["y"] = xyz[:, 1]
    arr["z"] = xyz[:, 2]
    arr["opacity"] = 2.0  # sigmoid(2.0) ~ 0.88
    arr["scale_0"] = np.log(0.1)
    arr["scale_1"] = np.log(0.1)
    arr["scale_2"] = np.log(0.1)
    arr["rot_0"] = 1.0
    splats.write_ply(path, splats.Ply(arr, []))


def test_asset_and_exe_names():
    assert visloc.asset_name("aarch64-apple-darwin") == f"visloc-rs-v{visloc.VERSION}-aarch64-apple-darwin.tar.gz"
    assert visloc.asset_name("x86_64-pc-windows-msvc") == f"visloc-rs-v{visloc.VERSION}-x86_64-pc-windows-msvc.zip"
    assert visloc.exe_name("x86_64-unknown-linux-gnu") == "gsplat_photos"
    assert visloc.exe_name("x86_64-pc-windows-msvc") == "gsplat_photos.exe"


def test_path_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom_gsplat_photos"
    monkeypatch.setenv("VISLOC", str(custom))
    assert visloc.path() == str(custom)
    monkeypatch.delenv("VISLOC", raising=False)
    assert visloc.path() == str(visloc.default_path())


def test_setup_verifies_sha256_and_installs(tmp_path):
    triple = "aarch64-apple-darwin"
    name = visloc.asset_name(triple)
    buf = io.BytesIO()
    payload = b"#!/bin/sh\necho gsplat_photos\n"
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(f"visloc-rs-v{visloc.VERSION}-{triple}/gsplat_photos")
        info.size = len(payload)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(payload))
    archive_bytes = buf.getvalue()
    digest = hashlib.sha256(archive_bytes).hexdigest()

    def fake_fetch(_url: str, dst: Path) -> None:
        if dst.name.endswith(".sha256"):
            dst.write_text(f"{digest}  {name}\n")
        else:
            dst.write_bytes(archive_bytes)

    exe = visloc.setup(install_dir=tmp_path, triple=triple, fetch=fake_fetch, pinned={name: digest})
    assert exe.is_file()
    assert exe.stat().st_mode & stat.S_IXUSR


def test_setup_rejects_bad_checksum(tmp_path):
    triple = "x86_64-unknown-linux-gnu"
    name = visloc.asset_name(triple)

    def fake_fetch(_url: str, dst: Path) -> None:
        if dst.name.endswith(".sha256"):
            dst.write_text(f"{'0' * 64}  {name}\n")
        else:
            dst.write_bytes(b"corrupt")

    with pytest.raises(ValueError):
        visloc.setup(install_dir=tmp_path, triple=triple, fetch=fake_fetch, pinned={name: "f" * 64})


def test_convert_colmap_bin_to_txt_and_rewrite_names(tmp_path):
    src = tmp_path / "bin"
    dst = tmp_path / "txt"
    _write_colmap_bin(src, ["front_0001.jpg", "front_wide_0001.jpg", "left_pillar_0001.jpg"])
    visloc.convert_colmap_bin_to_txt(src, dst)

    assert "1 PINHOLE 1280 960 900 900 640 480" in (dst / "cameras.txt").read_text()
    assert (dst / "points3D.txt").read_text().splitlines()[-1].startswith("1 0 0 2 128 128 128 0.5")

    visloc._rewrite_image_names(dst / "images.txt", ["front", "front_wide", "left_pillar"])
    lines = [ln for ln in (dst / "images.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    names = [ln.rsplit(" ", 1)[1] for ln in lines[::2]]
    assert names == [
        "front/front_0001.jpg",
        "front_wide/front_wide_0001.jpg",
        "left_pillar/left_pillar_0001.jpg",
    ]


def test_stage_inputs_flattens_images_and_masks(tmp_path):
    work = tmp_path / "run"
    for cam in ("front", "back"):
        (work / "images" / cam).mkdir(parents=True)
        (work / "masks" / cam).mkdir(parents=True)
        (work / "images" / cam / "0001.jpg").write_bytes(b"jpg")
        (work / "masks" / cam / "0001.jpg.png").write_bytes(b"png")

    img_dir, mask_dir = visloc.stage_inputs(work)
    assert sorted(p.name for p in img_dir.iterdir()) == ["back_0001.jpg", "front_0001.jpg"]
    assert sorted(p.name for p in mask_dir.iterdir()) == ["back_0001.jpg.png", "front_0001.jpg.png"]


def test_visloc_run_and_cli_recon(monkeypatch, tmp_path):
    work = tmp_path / "work" / "demo"
    for cam in ("front", "back"):
        (work / "images" / cam).mkdir(parents=True)
        (work / "masks" / cam).mkdir(parents=True)
        for i in range(1, 7):
            (work / "images" / cam / f"000{i}.jpg").write_bytes(b"jpg")
            (work / "masks" / cam / f"000{i}.jpg.png").write_bytes(b"png")
    (work / "meta.json").write_text(
        json.dumps({"cams": ["front", "back"], "fps": 2.0, "dur": 3.0, "start": 0.0, "clip": "2026-01-01_12-00-00"})
    )
    (work / "viewer").mkdir(parents=True, exist_ok=True)
    (work / "viewer" / "telemetry.json").write_text(json.dumps([{"t": i * 0.5, "mph": 25.0} for i in range(8)]))
    # Stale COLMAP artifacts from a prior run must be cleared when switching to visloc.
    (work / "gps_priors.json").write_text(json.dumps({"priors": 5}))
    (work / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (work / "sparse" / "0" / "stale.txt").write_text("stale")

    # Stub gsplat_photos binary via a helper script in the test dir
    stub = tmp_path / "stub_visloc.py"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import argparse, pathlib, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "from test_visloc import _write_colmap_bin, _write_minimal_splat_ply\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--images', required=True)\n"
        "ap.add_argument('--masks', required=True)\n"
        "ap.add_argument('--out', required=True)\n"
        "ap.add_argument('--steps', type=int, default=15000)\n"
        "args = ap.parse_args()\n"
        "out = pathlib.Path(args.out)\n"
        "imgs = sorted(p.name for p in pathlib.Path(args.images).glob('*.jpg'))\n"
        "_write_colmap_bin(out / 'sparse' / '0', imgs)\n"
        "_write_minimal_splat_ply(out / 'scene.ply')\n"
    )
    stub.chmod(0o755)
    monkeypatch.setenv("VISLOC", str(stub))
    monkeypatch.setattr(cli, "WORK", tmp_path / "work")

    preflight_skips = []
    monkeypatch.setattr(doctor, "preflight", lambda step, skip=(): preflight_skips.append((step, skip)))

    cli.main(["recon", "demo", "--backend", "visloc", "--steps", "500"])

    assert preflight_skips == [("recon", ("colmap", "brush"))]
    rargs = json.loads((work / "recon_args.json").read_text())
    assert rargs["backend"] == "visloc"
    assert rargs["steps"] == 500
    assert (work / "dense" / "sparse" / "cameras.txt").is_file()
    assert (work / "dense" / "sparse" / "images.txt").is_file()
    assert (work / "dense" / "images" / "front" / "front_0001.jpg").is_file()
    assert (work / "viewer" / "path.json").is_file()
    assert (work / "viewer" / "scene.ply").is_file()
    assert (work / "viewer" / "scene_raw.ply").is_file()
    assert (work / "viewer" / "collision.json").is_file()
    assert not (work / "gps_priors.json").exists()
    assert not (work / "sparse" / "0" / "stale.txt").exists()
    assert (work / "sparse" / "0" / "images.bin").is_file()
    path_doc = json.loads((work / "viewer" / "path.json").read_text())
    assert path_doc["scaleSource"] != "gps-priors"
    from tsplat import bench

    assert bench._sfm_model(work) == work / "dense" / "sparse"

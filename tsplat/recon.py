"""Structure-from-motion (COLMAP, CPU) -> undistorted dataset -> Brush splat training (Metal)
-> ego trajectory for the viewer's chase camera."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

BRUSH = os.environ.get("BRUSH", str(Path.home() / ".local/opt/brush/brush-app-aarch64-apple-darwin/brush_app"))


def sh(*args: str) -> None:
    print("$", " ".join(args[:2]), "...")
    subprocess.run(list(args), check=True)


def _registered(model: Path) -> int:
    return (model / "images.bin").stat().st_size if (model / "images.bin").exists() else 0


def _map(work: Path, db: Path, out: Path, global_mapper: bool, *extra: str) -> Path | None:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    sh(
        "colmap",
        "global_mapper" if global_mapper else "mapper",
        "--database_path",
        str(db),
        "--image_path",
        str(work / "images"),
        "--output_path",
        str(out),
        *extra,
    )
    models = [d for d in out.iterdir() if _registered(d)]
    return max(models, key=_registered) if models else None


def sfm(work: Path, global_mapper: bool = False, rig: bool = True) -> Path:
    """Features -> rig-aware pairs -> mapping. With rig=True the 6 cameras are tied together as one
    rigid rig (approximate Tesla layout as a prior), which keeps all cameras in a single model."""
    db = work / "colmap.db"
    if db.exists():
        db.unlink()
    sh(
        "colmap",
        "feature_extractor",
        "--database_path",
        str(db),
        "--image_path",
        str(work / "images"),
        "--ImageReader.camera_model",
        "OPENCV",
        "--ImageReader.single_camera_per_folder",
        "1",
        "--ImageReader.mask_path",
        str(work / "masks"),
        "--FeatureExtraction.use_gpu",
        "0",
    )
    if rig:
        sh("colmap", "rig_configurator", "--database_path", str(db), "--rig_config_path", str(work / "rig_config.json"))
    sh(
        "colmap",
        "matches_importer",
        "--database_path",
        str(db),
        "--match_list_path",
        str(work / "pairs.txt"),
        "--match_type",
        "pairs",
        "--FeatureMatching.use_gpu",
        "0",
    )
    best = _map(work, db, work / "sparse", global_mapper)
    if best is None:
        raise SystemExit(
            "COLMAP could not register the images. Try a window where the car is moving "
            "(10-40 mph), a higher --fps, --global-mapper, or --no-rig."
        )
    print(f"using model {best} ({_registered(best)} bytes of poses)")
    return best


def undistort(work: Path, model: Path) -> Path:
    """Undistort to PINHOLE, then lay out for Brush: unique stems (<cam>/<cam>_00000.jpg),
    TXT model, masks in dense/images/masks/<cam>_00000.png."""
    dense = work / "dense"
    shutil.rmtree(dense, ignore_errors=True)
    sh(
        "colmap",
        "image_undistorter",
        "--image_path",
        str(work / "images"),
        "--input_path",
        str(model),
        "--output_path",
        str(dense),
        "--output_type",
        "COLMAP",
    )
    stage, tmp = work / "_maskimgs", work / "_maskdense"
    for d in (stage, tmp):
        shutil.rmtree(d, ignore_errors=True)
    for m in (work / "masks").rglob("*.jpg.png"):
        out = stage / m.parent.name / m.name[:-4]
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.open(m).convert("RGB").save(out, quality=100)
    sh(
        "colmap",
        "image_undistorter",
        "--image_path",
        str(stage),
        "--input_path",
        str(model),
        "--output_path",
        str(tmp),
        "--output_type",
        "COLMAP",
    )
    md = dense / "images" / "masks"
    md.mkdir(parents=True, exist_ok=True)
    for f in (tmp / "images").rglob("*.jpg"):
        a = np.asarray(Image.open(f).convert("L"))
        Image.fromarray(np.where(a > 127, 255, 0).astype(np.uint8)).save(md / f"{f.parent.name}_{f.stem}.png")
    shutil.rmtree(stage)
    shutil.rmtree(tmp)

    for f in list((dense / "images").glob("*/*.jpg")):
        if f.parent.name != "masks":
            f.rename(f.parent / f"{f.parent.name}_{f.name}")
    sp = dense / "sparse"
    sh("colmap", "model_converter", "--input_path", str(sp), "--output_path", str(sp), "--output_type", "TXT")
    for b in sp.glob("*.bin"):
        b.unlink()
    lines = (sp / "images.txt").read_text().splitlines()
    out_lines, n = [], 0
    for line in lines:
        if not line.startswith("#"):  # count empty POINTS2D lines too, or pose/points parity drifts
            n += 1
            if n % 2 == 1:  # pose line: ... CAMERA_ID NAME
                head, name = line.rsplit(" ", 1)
                cam, fn = name.split("/", 1)
                line = f"{head} {cam}/{cam}_{fn}"
        out_lines.append(line)
    (sp / "images.txt").write_text("\n".join(out_lines) + "\n")
    for extra in ("rigs.txt", "frames.txt"):  # COLMAP 4 extras Brush doesn't use
        (sp / extra).unlink(missing_ok=True)
    _ensure_min_size(dense)
    return dense


def _ensure_min_size(dense: Path, min_side: int = 360) -> None:
    """Brush 0.3 fails ("early eof") on very small images. Heavy undistortion can shrink a
    camera's frames, so upscale those cameras (images, masks, intrinsics) to a safe size."""
    sp = dense / "sparse" / "cameras.txt"
    names = {}
    for line in (dense / "sparse" / "images.txt").read_text().splitlines():
        p = line.split()
        if len(p) == 10 and not line.startswith("#"):
            names.setdefault(p[8], []).append(p[9])
    out = []
    for line in sp.read_text().splitlines():
        p = line.split()
        if line.startswith("#") or len(p) < 8 or p[1] != "PINHOLE" or min(int(p[2]), int(p[3])) >= min_side:
            out.append(line)
            continue
        s = int(np.ceil(min_side / min(int(p[2]), int(p[3]))))
        w, h = int(p[2]) * s, int(p[3]) * s
        fx, fy, cx, cy = (float(v) * s for v in p[4:8])
        out.append(f"{p[0]} PINHOLE {w} {h} {fx} {fy} {cx} {cy}")
        for name in names.get(p[0], []):
            img = dense / "images" / name
            Image.open(img).resize((w, h), Image.LANCZOS).save(img, quality=95)
            m = dense / "images" / "masks" / f"{Path(name).stem}.png"
            if m.exists():
                Image.open(m).resize((w, h), Image.NEAREST).save(m)
        print(f"camera {p[0]}: upscaled x{s} for Brush (small undistorted frames)")
    sp.write_text("\n".join(out) + "\n")


def train(work: Path, dense: Path, steps: int, viewer: bool, eval_every: int = 0) -> Path:
    """Train the splat. eval_every=N holds out every Nth view and saves its renders for `otrv bench`."""
    out = work / "splat"
    out.mkdir(exist_ok=True)
    args = [
        BRUSH,
        str(dense),
        "--total-steps",
        str(steps),
        "--export-path",
        str(out),
        "--export-every",
        str(max(1000, steps // 4)),
        "--export-name",
        "splat_{iter}.ply",
    ]
    if viewer:
        args.append("--with-viewer")
    if eval_every:
        args += ["--eval-split-every", str(eval_every), "--eval-every", str(steps), "--eval-save-to-disk"]
    sh(*args)
    final = max(out.glob("splat_*.ply"), key=lambda p: int(p.stem.split("_")[1]))
    dest = work / "viewer" / "scene.ply"
    shutil.copy2(final, dest)
    print(f"splat -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def qvec_to_R(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _image_lines(dense: Path) -> list[str]:
    """Non-comment lines of images.txt, alternating pose / POINTS2D. Empty POINTS2D lines (images
    with no observations) are kept so that [::2] always selects pose lines."""
    return [line for line in (dense / "sparse" / "images.txt").read_text().splitlines() if not line.startswith("#")]


def export_path(work: Path, dense: Path) -> Path:
    """viewer/path.json: the car's trajectory per sampled frame (mean of registered camera centres,
    smoothed), heading from the trajectory tangent, scene up, and metres-per-unit from SEI speed."""
    meta = json.loads((work / "meta.json").read_text())
    lines = _image_lines(dense)
    centres: dict[int, list] = {}
    ups: dict[int, list] = {}
    # Follow the front camera (rig reference, CAM_HEIGHT above the road) when it is registered.
    # Averaging all cameras pulls the point down/back (the back camera is ~3 m behind and lower) and
    # tilts "up" (the back camera is pitched down), which sank the car into the road.
    poses = [line.split() for line in lines[::2]]
    ref = "front" if any(p[9].startswith("front/") for p in poses) else None
    for p in poses:
        if ref and not p[9].startswith(ref + "/"):
            continue
        R = qvec_to_R([float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        idx = int(Path(p[9]).stem.split("_")[-1])
        centres.setdefault(idx, []).append(-R.T @ t)
        ups.setdefault(idx, []).append(R.T @ np.array([0, -1.0, 0]))
    if len(centres) < 3:
        raise SystemExit("too few registered frames to build a camera path")
    idxs = sorted(centres)
    P = np.array([np.mean(centres[i], 0) for i in idxs])
    k = 2  # moving-average smoothing
    Ps = np.array([P[max(0, i - k) : i + k + 1].mean(0) for i in range(len(P))])
    F = np.gradient(Ps, axis=0)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    # up = camera up with pitch removed (perpendicular to the direction of travel)
    U = np.array([np.mean(ups[i], 0) for i in idxs])
    U -= (U * F).sum(1, keepdims=True) * F
    up = U.mean(0)
    up /= np.linalg.norm(up)
    F -= np.outer(F @ up, up)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    path = [
        {"t": round(i / meta["fps"], 3), "p": p.round(5).tolist(), "f": f.round(5).tolist()}
        for i, p, f in zip(idxs, Ps, F, strict=False)
    ]

    scale = 1.0
    tel = json.loads((work / "viewer" / "telemetry.json").read_text())
    units = float(np.linalg.norm(np.diff(Ps, axis=0), axis=1).sum())
    if tel:
        metres = sum(s["mph"] / 2.23694 * 0.1 for s in tel)  # 10 Hz samples
        if units > 1e-6 and metres > 5:
            scale = metres / units
    else:  # no SEI: assume ~11 m/s (25 mph) so the chase cam distance is sane
        scale = 11.0 * meta["dur"] / units if units > 1e-6 else 1.0
    out = work / "viewer" / "path.json"
    out.write_text(
        json.dumps(
            {
                "up": up.round(5).tolist(),
                "metresPerUnit": scale,
                "frames": path,
                "dur": meta["dur"],
                "cams": meta["cams"],
                "clip": meta["clip"],
                "timestamp": meta.get("timestamp"),
            }
        )
    )
    print(f"path: {len(path)} poses, {scale:.3f} m/unit -> {out}")
    return out


def export_nvs(work: Path, fps: float = 10.0, back_m: float = 9.0, up_m: float = 4.2, ahead_m: float = 14.0) -> Path:
    """Package posed reference frames + the chase-camera trajectory for a novel-view-synthesis model
    (e.g. ViewNeXt once its weights are public). Generic format: OpenCV camera convention
    (x right, y down, z forward), 4x4 camera-to-world matrices, pinhole intrinsics in pixels."""
    dense = work / "dense"
    path = json.loads((work / "viewer" / "path.json").read_text())
    up = np.array(path["up"])
    u = 1.0 / path["metresPerUnit"]
    cams = {}
    for line in (dense / "sparse" / "cameras.txt").read_text().splitlines():
        p = line.split()
        if p and not line.startswith("#"):
            cams[p[0]] = {
                "w": int(p[2]),
                "h": int(p[3]),
                "fx": float(p[4]),
                "fy": float(p[5]),
                "cx": float(p[6]),
                "cy": float(p[7]),
            }
    refs = []
    lines = _image_lines(dense)
    for line in lines[::2]:
        p = line.split()
        R = qvec_to_R([float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
        name = p[9]
        refs.append(
            {
                "image": str((dense / "images" / name).resolve()),
                "camera": name.split("/")[0],
                "t": int(Path(name).stem.split("_")[-1]) / json.loads((work / "meta.json").read_text())["fps"],
                "intrinsics": cams[p[8]],
                "c2w": c2w.round(6).tolist(),
            }
        )

    fr = path["frames"]
    ts = np.array([f["t"] for f in fr])
    P = np.array([f["p"] for f in fr])
    F = np.array([f["f"] for f in fr])
    traj = []
    for t in np.arange(ts[0], ts[-1], 1.0 / fps):
        p = np.array([np.interp(t, ts, P[:, k]) for k in range(3)])
        f = np.array([np.interp(t, ts, F[:, k]) for k in range(3)])
        f /= np.linalg.norm(f)
        road = p - up * 1.4 * u
        eye = road - f * back_m * u + up * up_m * u
        z = road + f * ahead_m * u + up * 1.0 * u - eye
        z /= np.linalg.norm(z)
        x = np.cross(-up, z)  # OpenCV: x = y × z with y ≈ down = -up
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
        traj.append({"t": round(float(t), 3), "c2w": c2w.round(6).tolist()})
    W, H, vfov = 960, 540, np.radians(55)  # same framing as the web viewer's chase cam
    fy = H / 2 / np.tan(vfov / 2)
    target_k = {"w": W, "h": H, "fx": fy, "fy": fy, "cx": W / 2, "cy": H / 2}
    out = work / "nvs"
    out.mkdir(exist_ok=True)
    (out / "views.json").write_text(
        json.dumps(
            {
                "convention": "opencv c2w (x right, y down, z forward); intrinsics in pixels",
                "metresPerUnit": path["metresPerUnit"],
                "up": path["up"],
                "references": refs,
                "target": {"intrinsics": target_k, "fps": fps, "frames": traj},
            }
        )
    )
    print(f"nvs: {len(refs)} posed references, {len(traj)} chase-cam target poses -> {out / 'views.json'}")
    return out

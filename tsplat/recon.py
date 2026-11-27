"""Structure-from-motion (COLMAP, CPU) -> undistorted dataset -> Brush splat training (Metal)
-> ego trajectory for the viewer's chase camera."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from . import brush, gps_priors
from .splats import CAM_HEIGHT_M


def sh(*args: str) -> None:
    print("$", " ".join(args[:2]), "...")
    subprocess.run(list(args), check=True)


def _registered(model: Path) -> int:
    """Number of registered images in a COLMAP model: images.bin starts with a little-endian uint64 count."""
    f = model / "images.bin"
    if not f.is_file():
        return 0
    with open(f, "rb") as fp:
        head = fp.read(8)
    return struct.unpack("<Q", head)[0] if len(head) == 8 else 0


def _map(
    work: Path,
    db: Path,
    out: Path,
    global_mapper: bool,
    *extra: str,
    pose_priors: bool = False,
) -> Path | None:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    mapper = "pose_prior_mapper" if pose_priors else ("global_mapper" if global_mapper else "mapper")
    cmd = [
        "colmap",
        mapper,
        "--database_path",
        str(db),
        "--image_path",
        str(work / "images"),
        "--output_path",
        str(out),
        *extra,
    ]
    if pose_priors:
        cmd += ["--use_robust_loss_on_prior_position", "1"]
    sh(*cmd)
    models = [d for d in out.iterdir() if _registered(d)]
    return max(models, key=_registered) if models else None


def _apply_gps_priors(work: Path, db: Path, sigma_h_m: float) -> bool:
    """Attach ENU GPS priors from viewer/telemetry.json to `colmap.db`. Returns True if applied."""
    tel_path = work / "viewer" / "telemetry.json"
    meta_path = work / "meta.json"
    if not tel_path.is_file() or not meta_path.is_file():
        print("gps-priors: skipped (missing telemetry.json or meta.json); running standard mapper")
        return False
    tel = json.loads(tel_path.read_text())
    meta = json.loads(meta_path.read_text())
    fps = float(meta.get("fps") or 0)
    cams = meta.get("cams") or []
    ref = "front" if "front" in cams else (cams[0] if cams else "front")
    frames = sorted((work / "images" / ref).glob("*.jpg"))
    if fps <= 0 or not frames:
        print("gps-priors: skipped (no reference frames); running standard mapper")
        return False
    frame_times = {int(f.stem): int(f.stem) / fps for f in frames}
    try:
        frame_enu, origin = gps_priors.sample_frame_enu(tel, frame_times)
        n = gps_priors.write_pose_priors(db, frame_enu, sigma_h_m=sigma_h_m, ref_cam=ref)
    except ValueError as e:
        print(f"gps-priors: skipped ({e}); running standard mapper")
        return False
    if n < 2:
        print(f"gps-priors: skipped (only {n} matching image priors); running standard mapper")
        return False
    (work / "gps_priors.json").write_text(
        json.dumps({"origin": [origin[0], origin[1]], "priors": n, "sigma_h_m": sigma_h_m})
    )
    print(f"gps-priors: wrote {n} ENU priors (sigma={sigma_h_m:g} m) to colmap.db")
    return True


def sfm(
    work: Path,
    global_mapper: bool = False,
    rig: bool = True,
    gps_priors: bool = False,
    gps_sigma_m: float = 4.0,
) -> Path:
    """Features -> rig-aware pairs -> mapping. With rig=True all cameras are tied together as one
    rigid rig (approximate Tesla layout as a prior), which keeps all cameras in a single model."""
    if gps_priors and global_mapper:
        raise SystemExit("--gps-priors uses COLMAP's pose_prior_mapper, which cannot be combined with --global-mapper")
    (work / "gps_priors.json").unlink(missing_ok=True)
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
    use_priors = _apply_gps_priors(work, db, gps_sigma_m) if gps_priors else False
    best = _map(work, db, work / "sparse", global_mapper, pose_priors=use_priors)
    if best is None:
        raise SystemExit(
            "COLMAP could not register the images. Try a window where the car is moving "
            "(10-40 mph), a higher --fps, --global-mapper, or --no-rig."
        )
    print(f"using model {best} ({_registered(best)} registered images)")
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
    _undistort_masks(work, model, work / "masks", dense / "images" / "masks")
    if (work / "sky").is_dir():  # bench scores without sky; `recon --sky` decides how Brush sees it
        _undistort_masks(work, model, work / "sky", dense / "sky")

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


def snapshot_dense(work: Path) -> None:
    """Keep the pristine undistorted dataset in dense_base/ so the training-time levers (sky, colour
    normalisation, dense init), which edit dense/ in place, start from it on every `recon --skip-sfm`."""
    base = work / "dense_base"
    shutil.rmtree(base, ignore_errors=True)
    shutil.copytree(work / "dense", base, ignore=shutil.ignore_patterns("stereo", "*.sh"))


def restore_dense(work: Path) -> Path:
    """dense/ as undistort left it (from dense_base/; a pre-v2 dense/ is pristine and becomes the base)."""
    dense, base = work / "dense", work / "dense_base"
    if not base.is_dir():
        snapshot_dense(work)
    else:
        shutil.rmtree(dense, ignore_errors=True)
        shutil.copytree(base, dense)
    return dense


def _undistort_masks(work: Path, model: Path, src: Path, dest: Path) -> None:
    """Undistort per-frame masks (<src>/<cam>/<frame>.jpg.png) with the images' cameras, by running
    them through image_undistorter as images; binarised to dest/<cam>_<frame>.png."""
    stage, tmp = work / "_maskimgs", work / "_maskdense"
    for d in (stage, tmp):
        shutil.rmtree(d, ignore_errors=True)
    for m in src.rglob("*.jpg.png"):
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
    dest.mkdir(parents=True, exist_ok=True)
    for f in (tmp / "images").rglob("*.jpg"):
        a = np.asarray(Image.open(f).convert("L"))
        Image.fromarray(np.where(a > 127, 255, 0).astype(np.uint8)).save(dest / f"{f.parent.name}_{f.stem}.png")
    shutil.rmtree(stage)
    shutil.rmtree(tmp)


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
            for m in (dense / "images" / "masks" / f"{Path(name).stem}.png", dense / "sky" / f"{Path(name).stem}.png"):
                if m.exists():
                    Image.open(m).resize((w, h), Image.NEAREST).save(m)
        print(f"camera {p[0]}: upscaled x{s} for Brush (small undistorted frames)")
    sp.write_text("\n".join(out) + "\n")


def export_every(steps: int) -> int:
    """Checkpoint interval that always lands on the final step, so splat_{steps}.ply is written."""
    return steps // 4 if steps % 4 == 0 and steps // 4 >= 1000 else steps


def train(
    work: Path,
    dense: Path,
    steps: int,
    viewer: bool,
    eval_every: int = 0,
    extra: list[str] | tuple = (),
    rules=None,
) -> Path:
    """Train the splat. eval_every=N holds out every Nth view and saves its renders for `otrv bench`;
    `extra` is passed through to Brush (e.g. --seed); `rules` (clean.Rules) prunes and dehazes the
    result on its way to viewer/scene.ply. splat/ is emptied first so a failed run can't leave an
    older run's splat or eval renders behind."""
    out = work / "splat"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    args = [
        brush.path(),
        str(dense),
        "--total-steps",
        str(steps),
        "--export-path",
        str(out),
        "--export-every",
        str(export_every(steps)),
        "--export-name",
        "splat_{iter}.ply",
        *extra,
    ]
    if viewer:
        args.append("--with-viewer")
    if eval_every:
        args += ["--eval-split-every", str(eval_every), "--eval-every", str(steps), "--eval-save-to-disk"]
    args, dropped = brush.adapt_args(args, brush.cli_flags(args[0]))  # v0.3.0 flags -> this Brush's
    if dropped:
        print(f"note: this Brush has no {', '.join(dropped)}; training without them")
    sh(*args)
    final = out / f"splat_{steps}.ply"
    if not final.exists():
        found = sorted(p.name for p in out.glob("splat_*.ply"))
        if not found:
            raise SystemExit(
                f"Brush exported nothing to {out}. Check its output above; `otrv doctor` checks the install."
            )
        raise SystemExit(
            f"Brush did not write the final splat {final.name} (found only {', '.join(found)}): "
            "training stopped early. Check its output above."
        )
    dest = work / "viewer" / "scene.ply"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for stale in (work / "clean.json", work / "viewer" / "scene_raw.ply", dest):
        stale.unlink(missing_ok=True)
    if rules and (rules.prune or rules.dehaze) and (work / "viewer" / "path.json").exists():
        from . import clean

        clean.run(work, final, rules)
    else:
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


_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapz was removed in NumPy 2.4


def metres_driven(tel: list[dict], t0: float, t1: float) -> float:
    """Distance from SEI speed integrated over the samples' real timestamps on [t0, t1] (speed is
    interpolated at the span ends), so it is right whatever the video/sample rate."""
    if len(tel) < 2 or t1 <= t0:
        return 0.0
    ts = np.array([s["t"] for s in tel], float)
    mps = np.array([s["mph"] for s in tel], float) / 2.23694
    order = np.argsort(ts)
    ts, mps = ts[order], mps[order]
    inner = (ts > t0) & (ts < t1)
    grid = np.concatenate([[t0], ts[inner], [t1]])
    return float(_trapezoid(np.interp(grid, ts, mps), grid))


def metres_per_unit(tel: list[dict], t0: float, t1: float, units: float) -> float:
    """Scene scale over the registered path's time span [t0, t1] (`units` = path length on that span)."""
    return scale_estimate(tel, t0, t1, units)[0]


def scale_estimate(tel: list[dict], t0: float, t1: float, units: float) -> tuple[float, str]:
    """Return the scale and its provenance; fallback values are not metric measurements."""
    if not np.isfinite([t0, t1, units]).all() or t1 <= t0 or units < 0:
        raise ValueError("scale requires a finite increasing time span and nonnegative path length")
    if units <= 1e-6:
        return 1.0, "default-stationary"
    if tel:
        metres = metres_driven(tel, t0, t1)
        if np.isfinite(metres) and metres > 5:
            return metres / units, "telemetry"
        return 1.0, "default-insufficient-telemetry"
    return 11.0 * (t1 - t0) / units, "assumed-speed"  # no SEI: assume ~11 m/s (25 mph)


def smooth_path(P: np.ndarray, k: int = 2) -> np.ndarray:
    """Centred moving average over 2k+1 samples. Near the ends the window shrinks symmetrically
    (radius min(k, i, n-1-i)), so the first and last samples stay where they were measured. A
    window truncated on one side only would pull them inward, which shortens the path and inflates
    the speed-derived metres-per-unit (by ~4% on 20 s drives at 3 fps)."""
    n = len(P)
    return np.array([P[i - r : i + r + 1].mean(0) for i in range(n) for r in [min(k, i, n - 1 - i)]])


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
    if not np.isfinite(P).all() or not np.isfinite(meta["fps"]) or meta["fps"] <= 0:
        raise SystemExit("camera path requires finite poses and a positive sample rate")
    Ps = smooth_path(P)
    F = np.gradient(Ps, axis=0)
    lengths = np.linalg.norm(F, axis=1)
    moving = np.flatnonzero(lengths > 1e-9)
    if not len(moving):
        raise SystemExit("camera path is stationary; select a clip with a moving camera")
    # Pauses retain the nearest measured travel direction instead of producing a zero heading.
    for i in np.flatnonzero(lengths <= 1e-9):
        F[i] = F[moving[np.argmin(np.abs(moving - i))]]
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    # up = camera up with pitch removed (perpendicular to the direction of travel)
    U = np.array([np.mean(ups[i], 0) for i in idxs])
    U -= (U * F).sum(1, keepdims=True) * F
    up = U.mean(0)
    up_length = np.linalg.norm(up)
    if not np.isfinite(up_length) or up_length <= 1e-9:
        raise SystemExit("camera path has no consistent up direction")
    up /= up_length
    F -= np.outer(F @ up, up)
    lengths = np.linalg.norm(F, axis=1, keepdims=True)
    if not np.isfinite(F).all() or (lengths <= 1e-9).any():
        raise SystemExit("camera path heading is parallel to its up direction")
    F /= lengths
    path = [
        {"t": round(i / meta["fps"], 3), "p": p.round(5).tolist(), "f": f.round(5).tolist()}
        for i, p, f in zip(idxs, Ps, F, strict=False)
    ]

    tel = json.loads((work / "viewer" / "telemetry.json").read_text())
    units = float(np.linalg.norm(np.diff(Ps, axis=0), axis=1).sum())
    if (work / "gps_priors.json").is_file():
        scale, scale_source = 1.0, "gps-priors"
    else:
        scale, scale_source = scale_estimate(tel, path[0]["t"], path[-1]["t"], units)
    cam_h = fit_cam_height(read_points(dense), Ps, up, scale)
    out = work / "viewer" / "path.json"
    doc = {
        "up": up.round(5).tolist(),
        "metresPerUnit": scale,
        "scaleSource": scale_source,
        "camHeight": round(cam_h if cam_h is not None else CAM_HEIGHT_M, 3),  # front camera above the road, m
        "camHeightSource": "sfm" if cam_h is not None else "default",
        "frames": path,
        "dur": meta["dur"],
        "cams": meta["cams"],
        "clip": meta["clip"],
        "timestamp": meta.get("timestamp"),
    }
    sky_json = work / "sky.json"
    colours = json.loads(sky_json.read_text()).get("colours") if sky_json.exists() else None
    if colours:  # mean real sky colour, sRGB 0-1: the viewer's background gradient
        doc["sky"] = {"top": colours["top"], "horizon": colours["horizon"]}
    out.write_text(json.dumps(doc, allow_nan=False))
    print(
        f"path: {len(path)} poses, {scale:.3f} m/unit ({scale_source}), camera {doc['camHeight']} m above the road "
        f"({doc['camHeightSource']}) -> {out}"
    )
    return out


def read_points(dense: Path) -> np.ndarray:
    """(n, 3) SfM points from dense/sparse/points3D.txt (empty if there is none)."""
    f = dense / "sparse" / "points3D.txt"
    if not f.exists():
        return np.zeros((0, 3))
    rows = [line.split()[1:4] for line in f.read_text().splitlines() if line.strip() and not line.startswith("#")]
    return np.array(rows, float).reshape(-1, 3)


def fit_cam_height(
    pts: np.ndarray, P: np.ndarray, up: np.ndarray, mpu: float, iters: int = 300, seed: int = 0
) -> float | None:
    """Front-camera height above the road in metres, from SfM points on the road beside the path.

    Candidates are points within 4 m (horizontally) of a path pose and 0.8-2.5 m below it. RANSAC fits
    a ribbon plane in path coordinates, drop = h + a * signed lateral offset (a = road camber or
    camera roll), with 5 cm inliers, then least squares on the inliers. Returns h, or None (use the
    1.4 m default) when fewer than 30 points support it or h is outside 1.0-2.2 m."""
    if len(pts) == 0 or len(P) < 2:
        return None
    up = up / np.linalg.norm(up)
    d = P[-1] - P[0]
    fwd = d - (d @ up) * up
    fwd /= np.linalg.norm(fwd) + 1e-12
    right = np.cross(fwd, up)
    H = np.stack([right, fwd], 1)

    _, idx = cKDTree(P @ H).query(pts @ H)
    rel = (pts - P[idx]) * mpu
    lateral = rel @ right
    lateral_abs = np.linalg.norm(rel @ H, axis=1)
    drop = -(rel @ up)
    cand = (lateral_abs <= 4.0) & (drop >= 0.8) & (drop <= 2.5)
    if cand.sum() < 30:
        return None
    s, y = lateral[cand], drop[cand]
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(s), 2, replace=False)
        if abs(s[i] - s[j]) < 0.2:
            continue
        a = (y[i] - y[j]) / (s[i] - s[j])
        h = y[i] - a * s[i]
        n = int((np.abs(h + a * s - y) < 0.05).sum())
        if abs(a) < 0.15 and n > best_n:
            best, best_n = (h, a), n
    if best is None or best_n < 30:
        return None
    inl = np.abs(best[0] + best[1] * s - y) < 0.05
    h, _ = np.linalg.lstsq(np.stack([np.ones(inl.sum()), s[inl]], 1), y[inl], rcond=None)[0]
    return float(h) if 1.0 <= h <= 2.2 else None


def export_nvs(work: Path, fps: float = 10.0, back_m: float = 9.0, up_m: float = 4.2, ahead_m: float = 14.0) -> Path:
    """Package posed reference frames + the chase-camera trajectory for a novel-view-synthesis model
    (e.g. ViewNeXt once its weights are public). Generic format: OpenCV camera convention
    (x right, y down, z forward), 4x4 camera-to-world matrices, pinhole intrinsics in pixels."""
    dense = work / "dense_base" if (work / "dense_base" / "sparse").is_dir() else work / "dense"  # no placeholders
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
        road = p - up * path.get("camHeight", CAM_HEIGHT_M) * u
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

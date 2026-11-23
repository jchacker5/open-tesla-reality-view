"""otrv: Tesla dashcam footage -> 3D Gaussian splat replay.

otrv doctor                                check tools (ffmpeg, COLMAP 4+, Brush) and print fixes
otrv setup-brush                           download + verify Brush v0.3.0 (the splat trainer)
otrv ingest [--src /Volumes/X/TeslaCam]    copy + index footage into the data dir
otrv list                                  clips with 4+ cameras and their mean speed
otrv suggest                               best 20 s windows of 10-40 mph driving, as prep commands
otrv egomask                               optional, once: mask your car's body (needs 8+ clips)
otrv prep <clip> --name drive1 --start 10 --dur 20
otrv recon drive1                          COLMAP -> Brush training -> chase path
otrv clean drive1                          re-run the post-train prune + dehaze with other thresholds
otrv collision drive1                      re-bake drive mode's ground + wall grids (recon does it too)
otrv view drive1                           open the replay viewer
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import importlib.util
import json
import os
import subprocess
import sys
import time
import webbrowser
from importlib import resources
from pathlib import Path

from . import bench, brush, clean, collision, colornorm, densify, doctor, ingest, prep, recon, serve, sky, speed

DEFAULT_DATA = "data/teslacam"
WORK = Path("work")
ROOT = Path(__file__).resolve().parent.parent


def viewer_html():
    """The viewer page: packaged in the wheel as tsplat/viewer/index.html, else the repo's viewer/."""
    packaged = resources.files("tsplat") / "viewer" / "index.html"
    if packaged.is_file():
        return packaged
    source = ROOT / "viewer" / "index.html"
    if not source.is_file():
        raise SystemExit("viewer/index.html is missing from this install; reinstall open-tesla-reality-view")
    return source


class Fmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults, except the uninformative ones (flags that default off, options with no default)."""

    def _get_help_string(self, action):
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)


def _kinds(value: str) -> tuple[str, ...]:
    kinds = tuple(k.strip() for k in value.split(",") if k.strip())
    bad = [k for k in kinds if k not in ingest.KINDS]
    if bad or not kinds:
        raise argparse.ArgumentTypeError(f"choose from {', '.join(ingest.KINDS)} (comma-separated), got {value!r}")
    return kinds


LPIPS_LOSS_WEIGHT = 0.05  # Brush's default is 0 (off); perceptual loss sharpens texture (bench: lpips, road)


def lpips_weight(requested: float | None, brush_version: tuple[int, int, int] | None) -> float:
    """--lpips-loss-weight, or the default for the installed Brush: 0.05, except 0 on Brush 0.x, whose
    LPIPS loss asks Metal for a 28 GB buffer (BufferTooBig) on images over ~256 px and aborts."""
    if requested is not None:
        return requested
    return LPIPS_LOSS_WEIGHT if brush_version and brush_version[0] >= 1 else 0.0


def default_max_splats() -> int:
    """Brush's splat cap: 3M on a 16 GB Mac (the trainer + macOS fit), 6M with more RAM (Brush's own is 10M)."""
    gb = doctor.ram_gb()
    return 6_000_000 if gb is not None and gb > 16.5 else 3_000_000


OPAC_LOSS_WEIGHT = 1e-9  # Brush default; 1e-7 is the first A/B (docs/HILL_CLIMB.md), 1e-6 collapsed the smoke scene
SCALE_LOSS_WEIGHT = 1e-8  # Brush's default, passed explicitly so it's recorded and A/B-able


def add_clean_args(a: argparse.ArgumentParser) -> None:
    """Flags for the post-train clean-up (clean.Rules), shared by `recon` and `clean`."""
    r = clean.Rules()
    a.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=r.prune,
        help="drop splats inside the car's swept path (1 m either side, 0.15-1.6 m up) and giant faint ones",
    )
    a.add_argument(
        "--dehaze",
        action=argparse.BooleanOptionalAction,
        default=r.dehaze,
        help="drop near-invisible, large translucent and needle splats and far ground fog (counts in clean.json)",
    )
    a.add_argument("--min-opacity", type=float, default=r.min_opacity, help="dehaze: drop splats below this opacity")
    a.add_argument(
        "--translucent-scale",
        type=float,
        default=r.translucent_scale_m,
        help="dehaze: metres; splats with a larger axis and opacity under --translucent-opacity go",
    )
    a.add_argument("--translucent-opacity", type=float, default=r.translucent_opacity, help="see --translucent-scale")
    a.add_argument(
        "--needle-ratio",
        type=float,
        default=r.needle_ratio,
        help="dehaze: largest/smallest axis above this with opacity under --needle-opacity is a needle",
    )
    a.add_argument("--needle-opacity", type=float, default=r.needle_opacity, help="see --needle-ratio")
    a.add_argument(
        "--ground-fog-m",
        type=float,
        default=r.ground_fog_m,
        help="dehaze: drop splats farther than this (metres) from the path within 1 m of road height",
    )
    a.add_argument(
        "--dehaze-sky",
        action="store_true",
        help="dehaze: also drop pale bright splats above eye level nearer than --sky-fog-m (sky painted at a "
        "finite depth); ignored when the sky was trained transparent (--sky alpha)",
    )
    a.add_argument("--sky-fog-m", type=float, default=r.sky_fog_m, help="see --dehaze-sky (metres)")
    a.add_argument("--sky-fog-saturation", type=float, default=r.sky_fog_saturation, help="see --dehaze-sky")
    a.add_argument("--sky-fog-brightness", type=float, default=r.sky_fog_brightness, help="see --dehaze-sky")


def clean_rules(args, sky_mode: str | None) -> clean.Rules:
    if args.dehaze_sky and sky_mode == "alpha":
        print("--dehaze-sky skipped: the sky was trained transparent (--sky alpha), so there is no sky fog to cut")
    return clean.Rules(
        prune=args.prune,
        dehaze=args.dehaze,
        min_opacity=args.min_opacity,
        translucent_scale_m=args.translucent_scale,
        translucent_opacity=args.translucent_opacity,
        needle_ratio=args.needle_ratio,
        needle_opacity=args.needle_opacity,
        ground_fog_m=args.ground_fog_m,
        sky_fog=args.dehaze_sky and sky_mode != "alpha",
        sky_fog_m=args.sky_fog_m,
        sky_fog_saturation=args.sky_fog_saturation,
        sky_fog_brightness=args.sky_fog_brightness,
    )


def build_parser() -> argparse.ArgumentParser:
    default_data = os.environ.get("OTRV_DATA", DEFAULT_DATA)
    ap = argparse.ArgumentParser(prog="otrv", description=__doc__, formatter_class=Fmt)
    ap.add_argument(
        "--data",
        default=default_data,
        help="footage folder used by ingest/list/suggest/egomask/prep (env: OTRV_DATA)",
    )
    ap.add_argument("--debug", action="store_true", help="show full Python tracebacks on errors")
    # The same two options are accepted after the subcommand too (`otrv list --data X`).
    data_opt = argparse.ArgumentParser(add_help=False)
    data_opt.add_argument(
        "--data",
        default=argparse.SUPPRESS,
        help=f"footage folder (default: $OTRV_DATA or {DEFAULT_DATA}; now {default_data})",
    )
    dbg = argparse.ArgumentParser(add_help=False)
    dbg.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help="show full tracebacks")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<command>")

    def cmd(name: str, help: str, data: bool = False) -> argparse.ArgumentParser:
        return sub.add_parser(
            name, help=help, description=help, formatter_class=Fmt, parents=[data_opt, dbg] if data else [dbg]
        )

    cmd("doctor", "check OS, ffmpeg, COLMAP 4.0+, Brush, disk and RAM; print the fix for each problem")

    a = cmd("setup-brush", f"download Brush v{brush.VERSION} for this platform, verify its sha256, install it")
    a.add_argument("--dir", default=str(brush.INSTALL_DIR), help="install directory")
    a.add_argument(
        "--from-source",
        action="store_true",
        help=f"instead, build Brush main at the pinned commit {brush.SOURCE_COMMIT[:12]} (needle-splat fix, "
        "mip filter, fisheye) with cargo into <dir>/src: 10-20 min and a few GB; v0.3.0 stays the default "
        "until you set BRUSH",
    )

    a = cmd("ingest", "copy TeslaCam footage off the USB drive into the data dir and index it", data=True)
    a.add_argument(
        "--src", help="TeslaCam folder or a folder of downloaded clips (default: auto-detect /Volumes/*/TeslaCam)"
    )
    a.add_argument("--dest", dest="data", default=argparse.SUPPRESS, help=argparse.SUPPRESS)  # old name of --data
    a.add_argument(
        "--kinds",
        type=_kinds,
        default=",".join(ingest.DEFAULT_KINDS),
        help="comma-separated folders to copy: RecentClips, SavedClips, SentryClips "
        "(Sentry clips are recorded parked, so they're opt-in)",
    )
    a.add_argument("--since", metavar="YYYY-MM-DD", help="only copy clips recorded on or after this date")
    a.add_argument(
        "--index-only",
        action="store_true",
        help="don't copy; just re-index the data dir (seconds per clip on first run to read speeds, then cached)",
    )

    a = cmd("list", "list indexed clips with camera count and mean speed", data=True)
    a.add_argument("--all", action="store_true", help="include clips with fewer than 4 cameras")

    a = cmd("suggest", "rank clip windows by sustained 10-40 mph driving and print `otrv prep` commands", data=True)
    a.add_argument("--dur", type=int, default=20, help="window length in seconds")
    a.add_argument("--min-mph", type=float, default=10, help="lowest speed that counts as good driving")
    a.add_argument("--max-mph", type=float, default=40, help="highest speed that counts as good driving")
    a.add_argument("--top", type=int, default=5, help="how many suggestions to print")

    a = cmd("egomask", "optional: build per-camera car-body masks from many clips (once; ~1-2 s per clip)", data=True)
    a.add_argument("--max-clips", type=int, default=40, help="clips to sample (one frame per camera from each)")

    a = cmd("prep", "extract frames, masks, rig prior and telemetry for a clip window (~1-2 min)", data=True)
    a.add_argument("clip", help="clip id or timestamp from `otrv list` / `otrv suggest`")
    a.add_argument("--name", required=True, help="run name; outputs go to work/<name>/")
    a.add_argument("--start", type=float, default=0, help="window start, in seconds into the clip")
    a.add_argument("--dur", type=float, default=20, help="window length in seconds (recon time grows with it)")
    a.add_argument(
        "--fps",
        type=float,
        default=6,
        help="frames per second sampled from each camera (20 s x 6 fps x 4 cams = 480); SfM time grows with it",
    )
    a.add_argument("--width", type=int, default=1280, help="max frame width in pixels for reconstruction")
    a.add_argument(
        "--mask-movers",
        action="store_true",
        help="mask cars/people with YOLO (needs `uv sync --extra movers`: ~1 GB torch download; ~0.2 s per frame)",
    )
    a.add_argument(
        "--sky-model",
        action="store_true",
        help="segment the sky with UperNet/ADE20K instead of the brightness heuristic "
        "(needs `uv sync --extra quality`; ~0.7 s per frame on an M4)",
    )

    a = cmd(
        "recon",
        "SfM (COLMAP) + splat training (Brush) + chase path. Most of the time is training: "
        "240 images at 30000 steps took 73 min on an M4 16 GB",
    )
    a.add_argument("name", help="run name from `otrv prep --name`")
    a.add_argument(
        "--steps",
        type=int,
        default=15000,
        help="Brush training steps; time is ~linear in steps (30000 steps on 240 images: 73 min on an M4 16 GB)",
    )
    a.add_argument("--global-mapper", action="store_true", help="use COLMAP's global mapper instead of incremental")
    a.add_argument(
        "--no-rig", action="store_true", help="don't tie the cameras together as a rigid rig (try if SfM fails)"
    )
    a.add_argument("--live", action="store_true", help="open Brush's live training viewer window")
    a.add_argument("--skip-sfm", action="store_true", help="reuse work/<name>/dense from the last recon; retrain only")
    a.add_argument(
        "--eval", action="store_true", help="hold out every 8th view (not trained on) so `otrv bench` can report PSNR"
    )
    a.add_argument("--seed", type=int, default=42, help="Brush random seed (fixed, so A/B runs differ by the lever)")
    a.add_argument(
        "--opac-loss-weight",
        type=float,
        default=OPAC_LOSS_WEIGHT,
        help="Brush opacity regulariser (Brush's own default 1e-9): higher kills translucent haze during training",
    )
    a.add_argument(
        "--scale-loss-weight",
        type=float,
        default=SCALE_LOSS_WEIGHT,
        help="Brush scale regulariser (Brush's own default 1e-8): higher discourages huge splats",
    )
    a.add_argument(
        "--max-splats",
        type=int,
        default=default_max_splats(),
        help="Brush's cap on the number of splats (default: 3M with 16 GB RAM or less, else 6M; Brush's own is 10M)",
    )
    a.add_argument(
        "--lpips-loss-weight",
        type=float,
        help=f"Brush perceptual (LPIPS) loss weight (default: {LPIPS_LOSS_WEIGHT} with a source-built Brush 1.x; "
        "0 with v0.3.0, whose LPIPS loss aborts on Metal for images over ~256 px; Brush's own default is 0)",
    )
    add_clean_args(a)
    a.add_argument(
        "--sky",
        choices=sky.MODES,
        default="alpha",
        help="alpha: sky transparent in training (ignored in images that also need an ego/mover mask); "
        "ignore: sky masked out; off: trained like the rest",
    )
    a.add_argument(
        "--dense-init",
        action="store_true",
        help="seed Brush with Depth Anything V2 depth fitted to the SfM points, plus a road ribbon "
        "(needs `uv sync --extra quality`; ~0.1 s per image on an M4)",
    )
    a.add_argument(
        "--color-norm",
        action="store_true",
        help="match every camera's colour to the front camera (LAB mean/std over a running 4 s window)",
    )

    a = cmd("clean", "re-run the post-train clean-up (ego-path prune + dehaze) on a trained splat, no retraining")
    a.add_argument("name", help="run name")
    add_clean_args(a)

    a = cmd(
        "collision",
        "bake viewer/collision.json for drive mode from the splat: ground height (0.5 m cells) and distance "
        "to walls (0.25 m cells) within 12 m of the path (seconds; recon and clean run it too)",
    )
    a.add_argument("name", help="run name")

    a = cmd("bench", "score a reconstruction, gate it, append it to work/ledger.jsonl (seconds)")
    a.add_argument("name", help="run name")
    a.add_argument("--note", default="", help="what changed in this run (the lever being tested)")

    a = cmd("nvs-export", "posed frames + chase-cam trajectory for a novel-view model (e.g. ViewNeXt)")
    a.add_argument("name", help="run name")
    a.add_argument("--fps", type=float, default=10, help="chase-cam target poses per second")

    a = cmd("view", "serve the replay viewer locally and open it (needs internet: three.js/Spark load from a CDN)")
    a.add_argument("name", help="run name")
    a.add_argument("--port", type=int, default=8765, help="local port")
    return ap


# ---- stage guards: say which step to run first instead of a traceback -------------------------------


def need_index(data: str) -> Path:
    idx = Path(data).expanduser() / "index.json"
    if not idx.exists():
        raise SystemExit(f"no clip index at {idx}: run `otrv ingest` first (or pass --data <folder>)")
    return idx


def need_prep(name: str) -> Path:
    work = WORK / name
    if not (work / "meta.json").exists():
        raise SystemExit(f"no prepared run {name!r} in {WORK}/: run `otrv prep <clip> --name {name}` first")
    return work


def need_recon(name: str, *files: str) -> Path:
    work = need_prep(name)
    missing = [f for f in files if not (work / f).exists()]
    if missing:
        raise SystemExit(f"{name} has no {missing[0]}: run `otrv recon {name}` first")
    return work


def recon_attempted(work: Path) -> bool:
    return any((work / f).exists() for f in ("colmap.db", "sparse", "dense", "splat", "recon_args.json"))


# ---- commands --------------------------------------------------------------------------------------


def cmd_list(args) -> None:
    clips = json.loads(need_index(args.data).read_text())
    shown = [c for c in clips if args.all or len(c["cams"]) >= 4]
    if not shown:
        print("no clips with 4+ cameras (use --all to see every clip)")
        return
    print(f"{'CLIP':<40} {'KIND':<12} {'LENGTH':>6}  CAMS  {'MPH':>4}  EVENT")
    for c in shown:
        reason = (c.get("event") or {}).get("reason", "")
        secs = f"{c['seconds']:.0f}s" if c.get("seconds") else "?"
        mph = f"{c['mph']:.0f}" if c.get("mph") is not None else "-"
        print(f"{c['id']:<40} {c['kind']:<12} {secs:>6}  {len(c['cams']):>4}  {mph:>4}  {reason}")
    if any("mph" not in c for c in shown):
        print("\n(MPH missing: re-run `otrv ingest --index-only` to read speeds from the clips' telemetry)")
    else:
        print("\nMPH: mean speed from the clip's telemetry; '-' = no telemetry (needs firmware 2025.44.25+)")


def cmd_suggest(args) -> None:
    clips = json.loads(need_index(args.data).read_text())
    picks = speed.suggest(clips, args.dur, args.min_mph, args.max_mph, args.top)
    if not picks:
        has = sum(1 for c in clips if c.get("mph_1hz"))
        why = (
            "no clip has telemetry speed (needs firmware 2025.44.25+; re-run `otrv ingest --index-only` "
            "if you ingested before this version)"
            if not has
            else f"no {args.dur} s window holds {args.min_mph:g}-{args.max_mph:g} mph in {has} clips with telemetry"
        )
        raise SystemExit(f"{why}. Pick a window by eye from `otrv list` instead.")
    data_flag = f" --data {args.data}" if args.data != DEFAULT_DATA else ""
    print(f"best {args.dur} s windows at {args.min_mph:g}-{args.max_mph:g} mph (Sentry clips excluded):\n")
    print(f"{'CLIP':<40} {'START':>5}  {'IN BAND':>7}  {'MEAN MPH':>8}")
    for p in picks:
        print(f"{p['id']:<40} {p['start']:>4}s  {p['share']:>7.0%}  {p['mean_mph']:>8.0f}")
    print("\nready to paste:")
    for p in picks:
        print(
            f"  otrv{data_flag} prep {p['id']} --name {speed.run_name(p['id'])} --start {p['start']} --dur {args.dur}"
        )


def cmd_recon(args) -> None:
    work = need_prep(args.name)
    if args.skip_sfm and not (work / "dense" / "sparse" / "images.txt").exists():
        raise SystemExit(f"--skip-sfm needs an earlier recon's {work / 'dense'}: run `otrv recon {args.name}` first")
    doctor.preflight("recon", skip=("colmap",) if args.skip_sfm else ())
    if args.dense_init and importlib.util.find_spec("transformers") is None:  # fail before SfM, not after
        raise SystemExit("--dense-init needs transformers: run `uv sync --extra quality` (~1 GB PyTorch download)")
    (work / "recon_args.json").unlink(missing_ok=True)  # written only when this run finishes
    dense = work / "dense"
    timings_path = work / "timings.json"
    timings = json.loads(timings_path.read_text()) if args.skip_sfm and timings_path.exists() else {}
    timings = {k: v for k, v in timings.items() if k in ("sfm", "undistort")}  # only the stages we reuse
    t0 = time.perf_counter()
    if not args.skip_sfm:
        model = recon.sfm(work, args.global_mapper, not args.no_rig)
        timings["sfm"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        dense = recon.undistort(work, model)
        recon.snapshot_dense(work)
        timings["undistort"] = time.perf_counter() - t0
    else:
        dense = recon.restore_dense(work)  # levers below edit dense/ in place; start each run clean
    recon.export_path(work, dense)
    if args.color_norm:  # before dense init, so the init points get the normalised colours
        t0 = time.perf_counter()
        colornorm.run(dense, json.loads((work / "meta.json").read_text())["fps"])
        timings["color_norm"] = time.perf_counter() - t0
    if args.dense_init:
        t0 = time.perf_counter()
        densify.run(work, dense)
        timings["dense_init"] = time.perf_counter() - t0
    sky.apply(dense, args.sky)  # last: it can turn images into RGBA .png and rename them
    t0 = time.perf_counter()
    brush_args = ["--seed", str(args.seed)]
    brush_args += [
        "--opac-loss-weight",
        f"{args.opac_loss_weight:g}",
        "--scale-loss-weight",
        f"{args.scale_loss_weight:g}",
        "--max-splats",
        str(args.max_splats),
    ]
    lpips = lpips_weight(args.lpips_loss_weight, brush.version(brush.path()))
    brush_args += ["--lpips-loss-weight", f"{lpips:g}"]
    rules = clean_rules(args, args.sky)
    recon.train(work, dense, args.steps, args.live, 8 if args.eval else 0, brush_args, rules)
    timings["train"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    collision.run(work)
    timings["collision"] = time.perf_counter() - t0
    timings_path.write_text(json.dumps({k: round(v, 1) for k, v in timings.items()}))
    (work / "recon_args.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "global_mapper": args.global_mapper,
                "rig": not args.no_rig,
                "eval": args.eval,
                "seed": args.seed,
                "opac_loss_weight": args.opac_loss_weight,
                "scale_loss_weight": args.scale_loss_weight,
                "max_splats": args.max_splats,
                "lpips_loss_weight": lpips,
                "sky": args.sky,
                "dense_init": args.dense_init,
                "color_norm": args.color_norm,
                "clean": dataclasses.asdict(rules),
            }
        )
    )
    print(f"done: `otrv view {args.name}` to replay it, `otrv bench {args.name}` to score it")


def cmd_clean(args) -> None:
    """Re-clean from the untouched trained splat (scene_raw.ply), so runs with different rules compare."""
    work = need_recon(args.name, "viewer/path.json")
    recon_args_path = work / "recon_args.json"
    rargs = json.loads(recon_args_path.read_text()) if recon_args_path.exists() else {}
    src = work / "viewer" / "scene_raw.ply"
    if not src.exists():
        src = work / "splat" / f"splat_{rargs.get('steps')}.ply"
    if not src.exists():
        raise SystemExit(f"{args.name} has no trained splat (viewer/scene_raw.ply): run `otrv recon {args.name}` first")
    rules = clean_rules(args, rargs.get("sky"))
    clean.run(work, src, rules)
    if (work / "viewer" / "path.json").exists():
        collision.run(work)  # walls and ground follow the splats that are left
    if rargs:  # the card's config should describe the scene that's now in viewer/
        rargs["clean"] = dataclasses.asdict(rules)
        recon_args_path.write_text(json.dumps(rargs))


def cmd_view(args) -> None:
    vd = need_recon(args.name, "viewer/scene.ply", "viewer/path.json") / "viewer"
    (vd / "index.html").write_bytes(viewer_html().read_bytes())
    try:
        srv = serve.make_server(vd, args.port)
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", -1)):
            raise SystemExit(
                f"port {args.port} is already in use (another `otrv view`?). "
                f"Try `otrv view {args.name} --port {args.port + 1}`."
            ) from e
        raise
    with srv:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"viewer: {url}  (ctrl-c to stop; needs internet for three.js/Spark from the CDN)")
        webbrowser.open(url)
        srv.serve_forever()


def cmd_setup_brush(args) -> None:
    if args.from_source:
        exe = brush.setup_from_source(Path(args.dir).expanduser())
        print(
            f"Brush main ({brush.SOURCE_COMMIT[:12]}) built: {exe}\n"
            f"recon keeps using {brush.path()} until you opt in:\n  export BRUSH='{exe}'\n"
            "(bench cards record the Brush version, so A/B it against v0.3.0 with the same run)"
        )
        return
    exe = brush.setup(Path(args.dir).expanduser())
    print(f"Brush installed: {exe}")
    if exe != brush.default_path() and os.environ.get("BRUSH") != str(exe):
        print(f"not the default location; add to your shell profile:\n  export BRUSH='{exe}'")


def dispatch(args) -> None:
    c = args.cmd
    if c == "doctor":
        sys.exit(doctor.doctor())
    elif c == "setup-brush":
        cmd_setup_brush(args)
    elif c == "ingest":
        doctor.preflight("ingest")
        ingest.run(args.src, args.data, args.index_only, args.kinds, args.since)
    elif c == "list":
        cmd_list(args)
    elif c == "suggest":
        cmd_suggest(args)
    elif c == "egomask":
        idx = need_index(args.data)
        doctor.preflight("egomask")
        prep.build_ego_masks(str(idx), str(idx.parent / "egomask"), args.max_clips)
    elif c == "prep":
        idx = need_index(args.data)
        doctor.preflight("prep")
        prep.run(
            str(idx),
            args.clip,
            args.name,
            args.start,
            args.dur,
            args.fps,
            args.width,
            args.mask_movers,
            sky_model=args.sky_model,
        )
    elif c == "recon":
        cmd_recon(args)
    elif c == "clean":
        cmd_clean(args)
    elif c == "collision":
        collision.run(need_recon(args.name, "viewer/scene.ply", "viewer/path.json"))
    elif c == "bench":
        work = need_prep(args.name)
        if not recon_attempted(work):
            raise SystemExit(f"{args.name} hasn't been reconstructed: run `otrv recon {args.name}` first")
        bench.run(args.name, args.note)
    elif c == "nvs-export":
        need_recon(args.name, "dense/sparse/images.txt", "viewer/path.json")
        recon.export_nvs(WORK / args.name, args.fps)
    elif c == "view":
        cmd_view(args)


def _cmd_str(cmd) -> str:
    parts = [cmd] if isinstance(cmd, str) else [str(x) for x in cmd]
    return " ".join([Path(parts[0]).name, *parts[1:2]])


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        dispatch(args)
    except KeyboardInterrupt:
        if args.debug:
            raise
        sys.exit("\ninterrupted")
    except subprocess.CalledProcessError as e:
        if args.debug:
            raise
        sys.exit(
            f"otrv: `{_cmd_str(e.cmd)}` failed (exit {e.returncode}); see its output above. "
            "Re-run with --debug for a traceback, or `otrv doctor` to check your tools."
        )
    except FileNotFoundError as e:
        if args.debug:
            raise
        sys.exit(f"otrv: not found: {e.filename or e}. `otrv doctor` checks your tools; --debug shows a traceback.")


if __name__ == "__main__":
    main()

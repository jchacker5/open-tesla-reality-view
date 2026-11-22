"""tsplat: Tesla dashcam -> 3D Gaussian splat replay.

tsplat ingest [--src /Volumes/X/TeslaCam]   copy + index footage into data/teslacam
tsplat list                                 show clips with 4+ cameras (6 on HW4)
tsplat egomask                              mask your car's body (once, needs >= 8 clips)
tsplat prep <clip> --name drive1 --start 10 --dur 20
tsplat recon drive1                         COLMAP -> Brush training -> chase path
tsplat view drive1                          open the replay viewer
"""

from __future__ import annotations

import argparse
import http.server
import json
import shutil
import socketserver
import webbrowser
from functools import partial
from pathlib import Path

from . import ingest, prep, recon

DATA = "data/teslacam"
ROOT = Path(__file__).resolve().parent.parent


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static server with HTTP Range support (browsers need it to seek <video>)."""

    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not rng.startswith("bytes=") or not Path(path).is_file():
            return super().send_head()
        size = Path(path).stat().st_size
        a, _, b = rng[6:].split(",")[0].partition("-")
        start = int(a) if a else max(0, size - int(b))
        end = min(int(b), size - 1) if a and b else size - 1
        if start >= size:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        n = getattr(self, "_remaining", None)
        if n is None:
            return super().copyfile(source, outputfile)
        while n > 0:
            chunk = source.read(min(65536, n))
            if not chunk:
                break
            outputfile.write(chunk)
            n -= len(chunk)
        self._remaining = None

    def log_message(self, *a):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="tsplat", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ingest", help="copy TeslaCam footage off the USB drive and index it")
    a.add_argument(
        "--src", help="TeslaCam folder or folder of downloaded clips (default: auto-detect /Volumes/*/TeslaCam)"
    )
    a.add_argument("--dest", default=DATA)
    a.add_argument("--index-only", action="store_true", help="just re-index --dest")

    a = sub.add_parser("list", help="list indexed clips")
    a.add_argument("--all", action="store_true", help="include clips with fewer than 4 cameras")

    a = sub.add_parser("egomask", help="build per-camera car-body masks from many clips (do once)")
    a.add_argument("--max-clips", type=int, default=40)

    a = sub.add_parser("prep", help="extract frames/masks/telemetry for a clip window")
    a.add_argument("clip", help="clip id or timestamp from `tsplat list`")
    a.add_argument("--name", required=True)
    a.add_argument("--start", type=float, default=0)
    a.add_argument("--dur", type=float, default=20)
    a.add_argument("--fps", type=float, default=3)
    a.add_argument("--width", type=int, default=1280)
    a.add_argument("--mask-movers", action="store_true", help="mask cars/people (needs `uv sync --extra movers`)")

    a = sub.add_parser("recon", help="SfM + splat training + chase path")
    a.add_argument("name")
    a.add_argument("--steps", type=int, default=15000)
    a.add_argument("--global-mapper", action="store_true")
    a.add_argument("--no-rig", action="store_true", help="don't tie the 6 cameras together as a rig")
    a.add_argument("--live", action="store_true", help="open Brush's live training viewer")
    a.add_argument("--skip-sfm", action="store_true", help="reuse work/<name>/dense")

    a = sub.add_parser("nvs-export", help="posed frames + chase-cam trajectory for a novel-view model (e.g. ViewNeXt)")
    a.add_argument("name")
    a.add_argument("--fps", type=float, default=10)

    a = sub.add_parser("view", help="serve the replay viewer")
    a.add_argument("name")
    a.add_argument("--port", type=int, default=8765)

    args = ap.parse_args()
    if args.cmd == "ingest":
        ingest.run(args.src, args.dest, args.index_only)
    elif args.cmd == "list":
        clips = json.loads((Path(DATA) / "index.json").read_text())
        for c in clips:
            if args.all or len(c["cams"]) >= 4:
                reason = (c.get("event") or {}).get("reason", "")
                print(f"{c['id']:<40} {c['kind']:<12} {c['seconds'] or '?':>6}s  {len(c['cams'])} cams  {reason}")
    elif args.cmd == "egomask":
        prep.build_ego_masks(str(Path(DATA) / "index.json"), str(Path(DATA) / "egomask"), args.max_clips)
    elif args.cmd == "prep":
        prep.run(
            str(Path(DATA) / "index.json"),
            args.clip,
            args.name,
            args.start,
            args.dur,
            args.fps,
            args.width,
            args.mask_movers,
        )
    elif args.cmd == "recon":
        work = Path("work") / args.name
        dense = work / "dense"
        if not args.skip_sfm:
            dense = recon.undistort(work, recon.sfm(work, args.global_mapper, not args.no_rig))
        recon.export_path(work, dense)
        recon.train(work, dense, args.steps, args.live)
    elif args.cmd == "nvs-export":
        recon.export_nvs(Path("work") / args.name, args.fps)
    elif args.cmd == "view":
        vd = Path("work") / args.name / "viewer"
        shutil.copy2(ROOT / "viewer" / "index.html", vd / "index.html")
        handler = partial(RangeHandler, directory=str(vd))
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("127.0.0.1", args.port), handler) as srv:
            url = f"http://127.0.0.1:{args.port}/"
            print(f"viewer: {url}  (ctrl-c to stop)")
            webbrowser.open(url)
            srv.serve_forever()


if __name__ == "__main__":
    main()

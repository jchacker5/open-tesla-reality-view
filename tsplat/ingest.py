"""Copy TeslaCam footage off the USB drive (or a folder of app downloads) and index it.

TeslaCam layout on the drive:
  TeslaCam/RecentClips/2026-11-20_07-20-00-front.mp4          (rolling last hour)
  TeslaCam/SavedClips/<event>/2026-11-20_07-19-00-front.mp4    (+ event.json, thumb.png)
  TeslaCam/SentryClips/<event>/...
Cameras: front, back, left_repeater, right_repeater, left_pillar, right_pillar (pillars: HW4).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

CAMS = ["front", "back", "left_repeater", "right_repeater", "left_pillar", "right_pillar"]
CLIP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})-(" + "|".join(CAMS) + r")\.mp4$", re.I)


def find_teslacam() -> Path | None:
    for vol in Path("/Volumes").iterdir():
        if (vol / "TeslaCam").is_dir():
            return vol / "TeslaCam"
    return None


def copy(src: Path, dest: Path) -> None:
    """rsync so re-running resumes and skips files already copied."""
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        subprocess.run(
            ["rsync", "-a", "--include=*/", "--include=*.mp4",
             "--include=event.json", "--exclude=*", f"{src}/", f"{dest}/"],
            check=True,
        )
    else:
        for f in src.rglob("*.mp4"):
            out = dest / f.relative_to(src)
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out)


def duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def build_index(root: Path) -> list[dict]:
    """Group per-camera files into clips keyed by (folder, timestamp)."""
    groups: dict[tuple[str, str], dict] = defaultdict(dict)
    for f in sorted(root.rglob("*.mp4")):
        m = CLIP_RE.search(f.name)
        if m:
            groups[(str(f.parent), m.group(1))][m.group(2).lower()] = f
    clips = []
    for (folder, ts), cams in sorted(groups.items(), key=lambda kv: kv[0][1]):
        kind = next((k for k in ("SavedClips", "SentryClips", "RecentClips") if k in folder), "Other")
        event = None
        ev = Path(folder) / "event.json"
        if ev.exists():
            try:
                event = json.loads(ev.read_text())
            except json.JSONDecodeError:
                pass
        clips.append({
            "id": f"{ts}" if kind == "RecentClips" else f"{Path(folder).name}/{ts}",
            "timestamp": ts,
            "kind": kind,
            "seconds": round(duration(cams["front"]), 2) if "front" in cams else None,
            "cams": {c: str(p) for c, p in sorted(cams.items(), key=lambda kv: CAMS.index(kv[0]))},
            "event": event,
        })
    return clips


def run(src: str | None, dest: str, index_only: bool = False) -> Path:
    dest_p = Path(dest).expanduser()
    if not index_only:
        src_p = Path(src).expanduser() if src else find_teslacam()
        if not src_p or not src_p.exists():
            raise SystemExit(
                "No TeslaCam folder found. Plug the car's USB drive into the Mac "
                "(it mounts under /Volumes/<name>/TeslaCam) or pass --src <folder>."
            )
        print(f"copying {src_p} -> {dest_p}")
        copy(src_p, dest_p)
    clips = build_index(dest_p)
    out = dest_p / "index.json"
    out.write_text(json.dumps(clips, indent=1))
    six = sum(1 for c in clips if len(c["cams"]) == 6)
    four = sum(1 for c in clips if len(c["cams"]) >= 4) - six
    print(f"indexed {len(clips)} clips ({six} with 6 cameras, {four} with 4-5) -> {out}")
    return out

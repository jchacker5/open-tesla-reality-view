"""Copy TeslaCam footage off the USB drive (or a folder of app downloads) and index it.

TeslaCam layout on the drive:
  TeslaCam/RecentClips/2026-11-20_07-20-00-front.mp4          (rolling last hour)
  TeslaCam/SavedClips/<event>/2026-11-20_07-19-00-front.mp4    (+ event.json, thumb.png)
  TeslaCam/SentryClips/<event>/...
Cameras: front, back, left_repeater, right_repeater, left_pillar, right_pillar (pillars: HW4).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from . import speed

CAMS = ["front", "back", "left_repeater", "right_repeater", "left_pillar", "right_pillar"]
CLIP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})-(" + "|".join(CAMS) + r")\.mp4$", re.I)
KINDS = ("RecentClips", "SavedClips", "SentryClips")
DEFAULT_KINDS = ("RecentClips", "SavedClips")  # Sentry clips are recorded parked: opt in with --kinds
HEADROOM = 1 << 30  # keep 1 GB free after copying


def find_teslacam() -> Path | None:
    vols = Path("/Volumes")  # macOS mount point; pass --src elsewhere
    for vol in vols.iterdir() if vols.is_dir() else []:
        if (vol / "TeslaCam").is_dir():
            return vol / "TeslaCam"
    return None


def parse_since(since: str | None) -> dt.date | None:
    if not since:
        return None
    try:
        return dt.date.fromisoformat(since)
    except ValueError as e:
        raise SystemExit(f"--since expects YYYY-MM-DD, got {since!r}") from e


def select_files(src: Path, kinds: tuple[str, ...] = DEFAULT_KINDS, since: dt.date | None = None) -> list[Path]:
    """Files to copy, relative to src: clip .mp4s (+ their event.json) in the chosen kinds, on/after `since`.
    A folder without RecentClips/SavedClips/SentryClips (e.g. app downloads) is taken whole."""
    roots = [src / k for k in kinds if (src / k).is_dir()]
    if not any((src / k).is_dir() for k in KINDS):
        roots = [src]
    files: list[Path] = []
    event_dirs: set[Path] = set()
    for root in roots:
        for f in sorted(root.rglob("*")):
            if f.name.startswith("._") or not f.is_file():
                continue
            m = CLIP_RE.match(f.name)
            if not m:
                continue
            if since and dt.date.fromisoformat(m.group(1)[:10]) < since:
                continue
            files.append(f.relative_to(src))
            event_dirs.add(f.parent)
    files += [(d / "event.json").relative_to(src) for d in sorted(event_dirs) if (d / "event.json").is_file()]
    return files


def bytes_to_copy(src: Path, dest: Path, files: list[Path]) -> tuple[int, int]:
    """(total bytes, bytes whose size or modification time differs at dest)."""
    total = new = 0
    for rel in files:
        size = (src / rel).stat().st_size
        total += size
        out = dest / rel
        if not _same_file_version(src / rel, out):
            new += size
    return total, new


def _same_file_version(source: Path, target: Path) -> bool:
    try:
        before, after = source.stat(), target.stat()
    except FileNotFoundError:
        return False
    return (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)


def free_bytes(path: Path) -> int:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def gb(n: int) -> str:
    return f"{n / 2**30:.1f} GB"


def rsync_progress_flag(version_text: str) -> str:
    """GNU rsync 3.1+ has a whole-transfer progress bar; macOS openrsync and older rsyncs only --progress."""
    if "openrsync" in version_text:
        return "--progress"
    m = re.search(r"version\s+(\d+)\.(\d+)", version_text)
    if m and (int(m.group(1)), int(m.group(2))) >= (3, 1):
        return "--info=progress2"
    return "--progress"


def copy(src: Path, dest: Path, files: list[Path]) -> None:
    """rsync the listed files so re-running resumes and skips files already copied."""
    dest.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        ver = subprocess.run([rsync, "--version"], capture_output=True, text=True).stdout
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
            lst.write("\n".join(str(f) for f in files) + "\n")
        try:
            subprocess.run(
                [
                    rsync,
                    "-a",
                    rsync_progress_flag(ver),
                    "--exclude=._*",
                    f"--files-from={lst.name}",
                    f"{src}/",
                    f"{dest}/",
                ],
                check=True,
            )
        finally:
            Path(lst.name).unlink(missing_ok=True)
        return
    for i, rel in enumerate(files, 1):
        out = dest / rel
        if not _same_file_version(src / rel, out):
            out.parent.mkdir(parents=True, exist_ok=True)
            # A cancelled copy must leave either the old complete file or the new complete file.
            with tempfile.NamedTemporaryFile(dir=out.parent, prefix=f".{out.name}.", delete=False) as tmp:
                staging = Path(tmp.name)
            try:
                shutil.copy2(src / rel, staging)
                staging.replace(out)
            finally:
                staging.unlink(missing_ok=True)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} files", flush=True)


def duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _previous(root: Path) -> dict[str, dict]:
    idx = root / "index.json"
    try:
        return {c["id"]: c for c in json.loads(idx.read_text())} if idx.exists() else {}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def build_index(root: Path, with_speed: bool = True) -> list[dict]:
    """Group per-camera files into clips keyed by (folder, timestamp). Paths are absolute, so the index
    works from any directory. Speed from the front camera's SEI is cached (re-used while the file is unchanged)."""
    root = root.expanduser().resolve()
    prev = _previous(root)
    groups: dict[tuple[str, str], dict] = defaultdict(dict)
    for f in sorted(p for p in root.rglob("*") if p.suffix.lower() == ".mp4"):
        if f.name.startswith("._"):  # macOS AppleDouble metadata files on exFAT/FAT drives
            continue
        m = CLIP_RE.match(f.name)
        if m:
            groups[(str(f.parent), m.group(1))][m.group(2).lower()] = f
    clips = []
    for (folder, ts), cams in sorted(groups.items(), key=lambda kv: kv[0][1]):
        kind = next((k for k in KINDS if k in folder), "Other")
        event = None
        ev = Path(folder) / "event.json"
        if ev.exists():
            try:
                event = json.loads(ev.read_text())
            except json.JSONDecodeError:
                pass
        cid = f"{ts}" if kind == "RecentClips" else f"{Path(folder).name}/{ts}"
        front = cams.get("front")
        clip = {
            "id": cid,
            "timestamp": ts,
            "kind": kind,
            "seconds": round(duration(front), 2) if front else None,
            "cams": {c: str(p) for c, p in sorted(cams.items(), key=lambda kv: CAMS.index(kv[0]))},
            "event": event,
        }
        if with_speed and front:
            stat = front.stat()
            size, mtime_ns = stat.st_size, stat.st_mtime_ns
            old = prev.get(cid, {})
            clip |= {"front_bytes": size, "front_mtime_ns": mtime_ns}
            if (
                old.get("cams", {}).get("front") == str(front)
                and old.get("front_bytes") == size
                and old.get("front_mtime_ns") == mtime_ns
                and "mph" in old
            ):
                clip |= {"mph": old["mph"], "mph_1hz": old.get("mph_1hz")}
            else:
                s = speed.clip_speed(str(front), clip["seconds"])
                clip |= {"mph": s and s["mean_mph"], "mph_1hz": s and s["mph_1hz"]}
        clips.append(clip)
    return clips


def run(
    src: str | None,
    dest: str,
    index_only: bool = False,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    since: str | None = None,
) -> Path:
    dest_p = Path(dest).expanduser()
    if not index_only:
        src_p = Path(src).expanduser() if src else find_teslacam()
        if not src_p or not src_p.exists():
            raise SystemExit(
                "No TeslaCam folder found. Plug the car's USB drive in (on a Mac it mounts under "
                "/Volumes/<name>/TeslaCam) or pass --src <folder>."
            )
        files = select_files(src_p, kinds, parse_since(since))
        if not files:
            raise SystemExit(
                f"no clips in {src_p} for --kinds {','.join(kinds)}" + (f" --since {since}" if since else "")
            )
        total, new = bytes_to_copy(src_p, dest_p, files)
        free = free_bytes(dest_p)
        print(f"source: {len(files)} files, {gb(total)} ({gb(new)} not yet copied); free at {dest_p}: {gb(free)}")
        if new + HEADROOM > free:
            raise SystemExit(
                f"not enough space: need {gb(new)} (+1 GB headroom) at {dest_p}, have {gb(free)}. "
                "Free some space, narrow with --since / --kinds, or use --data on a bigger disk."
            )
        print(f"copying {src_p} -> {dest_p}")
        copy(src_p, dest_p, files)
    if not dest_p.is_dir():
        raise SystemExit(f"{dest_p} doesn't exist; run `otrv ingest --src <TeslaCam folder>` first")
    print("indexing (reads speed from each clip's telemetry; cached after the first run)...")
    clips = build_index(dest_p)
    out = dest_p / "index.json"
    out.write_text(json.dumps(clips, indent=1))
    six = sum(1 for c in clips if len(c["cams"]) == 6)
    four = sum(1 for c in clips if len(c["cams"]) >= 4) - six
    print(f"indexed {len(clips)} clips ({six} with 6 cameras, {four} with 4-5) -> {out}")
    return out

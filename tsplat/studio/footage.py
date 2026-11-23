"""Footage for the studio: sources (TeslaCam folders indexed in place, never copied wholesale), clips,
drives (consecutive clips grouped), thumbnails, and the per-job copy of one clip's camera files.

Studio state lives in <data>/studio/: sources.json (folders you added), cache.json (per-clip length,
speed and GPS route, keyed by file path + size so a re-scan only reads new clips) and thumbs/.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import threading
from collections import defaultdict
from pathlib import Path

from .. import ingest, sei, speed

DRIVE_GAP = 10.0  # seconds between one clip's end and the next clip's start that still count as one drive
ROUTE_POINTS = 160  # max points per drive route polyline
SPEED_POINTS = 240  # max points per drive speed series sent to the grid (detail gets full 1 Hz)


def key_for(folder: str, ts: str) -> str:
    return hashlib.sha1(f"{folder}|{ts}".encode()).hexdigest()[:12]


def parse_ts(ts: str) -> dt.datetime:
    return dt.datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S")


def autodetect() -> list[Path]:
    """TeslaCam folders on mounted drives (macOS: /Volumes/*/TeslaCam)."""
    vols = Path("/Volumes")
    if not vols.is_dir():
        return []
    out = []
    for vol in sorted(vols.iterdir()):
        try:
            if (vol / "TeslaCam").is_dir():
                out.append(vol / "TeslaCam")
        except OSError:
            continue
    return out


def pick_folder() -> str | None:
    """macOS folder chooser via osascript. None if cancelled or unavailable."""
    if not shutil.which("osascript"):
        return None
    r = subprocess.run(
        ["osascript", "-e", 'POSIX path of (choose folder with prompt "Choose your TeslaCam folder")'],
        capture_output=True,
        text=True,
        timeout=600,
    )
    path = r.stdout.strip().rstrip("/")
    return path if r.returncode == 0 and path else None


def clip_info(front: Path, seconds: float | None) -> dict:
    """Speed (1 Hz) and GPS route (1 Hz, [lat, lon]) from the front camera's SEI telemetry."""
    try:
        rows = sei.extract(front)
    except (OSError, RuntimeError, ValueError):
        rows = []
    s = speed.summarize(rows, seconds)
    route: list[list[float]] = []
    if rows and seconds:
        step = max(1, int(len(rows) / max(1.0, seconds)))
        for r in rows[::step]:
            lat, lon = r.get("latitude_deg"), r.get("longitude_deg")
            if lat and lon and abs(lat) <= 90 and abs(lon) <= 180:
                route.append([round(lat, 6), round(lon, 6)])
    return {"mph": s and s["mean_mph"], "mph_1hz": s and s["mph_1hz"], "route": route}


def downsample(seq: list, n: int) -> list:
    if len(seq) <= n:
        return seq
    step = len(seq) / n
    return [seq[int(i * step)] for i in range(n)] + [seq[-1]]


class Library:
    """Sources -> clips -> drives. Scanning runs on a background thread; reads are served from memory."""

    def __init__(self, data: Path, autodetect_fn=autodetect):
        self.data = Path(data).expanduser()
        self.state = self.data / "studio"
        self.thumbs = self.state / "thumbs"
        self.uploads = self.data / "uploads"
        self.autodetect_fn = autodetect_fn
        self.lock = threading.RLock()
        self.clips: dict[str, dict] = {}  # key -> clip
        self.scan_state = {"running": False, "done": 0, "total": 0, "source": None, "error": None}
        self._cache: dict[str, dict] = self._read(self.state / "cache.json", {})
        self._thread: threading.Thread | None = None
        self._rescan = False

    # ---- persistence ----

    @staticmethod
    def _read(path: Path, default):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return default

    def _write(self, name: str, obj) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        tmp = self.state / f"{name}.tmp"
        tmp.write_text(json.dumps(obj))
        tmp.replace(self.state / name)

    # ---- sources ----

    def manual_sources(self) -> list[str]:
        return self._read(self.state / "sources.json", [])

    def sources(self) -> list[dict]:
        """Every place clips come from: USB drives (auto), folders you added, the ingest data dir, uploads."""
        seen: set[str] = set()
        out: list[dict] = []

        def add(path: Path, kind: str, removable: bool) -> None:
            p = str(path.expanduser().resolve()) if path.exists() else str(path.expanduser())
            if p in seen:
                return
            seen.add(p)
            with self.lock:
                n = sum(1 for c in self.clips.values() if c["source"] == p)
            out.append({"path": p, "kind": kind, "exists": Path(p).is_dir(), "clips": n, "removable": removable})

        for p in self.autodetect_fn():
            add(p, "usb", False)
        for p in self.manual_sources():
            add(Path(p), "folder", True)
        if self.data.is_dir() and any(self.data.glob("*Clips")):
            add(self.data, "data", False)
        if self.uploads.is_dir():
            add(self.uploads, "uploads", False)
        return out

    def add_source(self, path: str) -> dict:
        p = Path(path).expanduser()
        if not p.is_dir():
            raise ValueError(f"{path} is not a folder")
        p = p.resolve()
        if not any(ingest.CLIP_RE.match(f.name) for f in _walk(p, limit=20000)):
            raise ValueError(f"no TeslaCam clips (e.g. 2026-11-20_07-20-00-front.mp4) under {p}")
        srcs = self.manual_sources()
        if str(p) not in srcs:
            srcs.append(str(p))
            self._write("sources.json", srcs)
        self.scan()
        return next(s for s in self.sources() if s["path"] == str(p))

    def remove_source(self, path: str) -> None:
        srcs = [s for s in self.manual_sources() if s != path]
        self._write("sources.json", srcs)
        with self.lock:
            self.clips = {k: c for k, c in self.clips.items() if c["source"] != path}

    # ---- scanning ----

    def scan(self, wait: bool = False) -> None:
        """(Re)index every source in the background. Clips already in the cache are not re-read."""
        with self.lock:
            if self._thread and self._thread.is_alive():
                self._rescan = True
            else:
                self._thread = threading.Thread(target=self._scan_loop, name="otrv-studio-scan", daemon=True)
                self.scan_state["running"] = True
                self._thread.start()
            t = self._thread
        if wait:
            t.join()

    def _scan_loop(self) -> None:
        while True:
            try:
                self._scan_all()
            except Exception as e:  # keep the server up; surface the error in /api/sources
                self.scan_state["error"] = f"{type(e).__name__}: {e}"
            with self.lock:
                if not self._rescan:
                    self.scan_state["running"] = False
                    return
                self._rescan = False

    def _scan_all(self) -> None:
        found: dict[str, dict] = {}
        groups: list[tuple[str, str, str, dict[str, Path]]] = []
        for src in self.sources():
            if not src["exists"]:
                continue
            for (folder, ts), cams in group_files(Path(src["path"])).items():
                groups.append((src["path"], folder, ts, cams))
        self.scan_state.update(done=0, total=len(groups), error=None)
        dirty = 0
        for i, (source, folder, ts, cams) in enumerate(groups, 1):
            clip = self._clip(source, folder, ts, cams)
            if clip.pop("_new", False):
                dirty += 1
            found[clip["key"]] = clip
            self.scan_state.update(done=i, source=source)
            if dirty and dirty % 10 == 0:
                self._write("cache.json", self._cache)
            with self.lock:
                self.clips[clip["key"]] = clip
        if dirty:
            self._write("cache.json", self._cache)
        with self.lock:
            self.clips = found

    def _clip(self, source: str, folder: str, ts: str, cams: dict[str, Path]) -> dict:
        kind = next((k for k in ingest.KINDS if k in folder), "Other")
        front = cams.get("front") or next(iter(cams.values()))
        st = front.stat()
        ck = f"{front}|{st.st_size}|{int(st.st_mtime)}"
        info = self._cache.get(ck)
        new = info is None
        if new:
            seconds = round(ingest.duration(front), 2) or None
            info = {"seconds": seconds, **(clip_info(front, seconds) if "front" in cams else {})}
            self._cache[ck] = info
        event = None
        ev = Path(folder) / "event.json"
        if ev.is_file():
            event = self._read(ev, None)
        cid = ts if kind == "RecentClips" else f"{Path(folder).name}/{ts}"
        return {
            "key": key_for(folder, ts),
            "id": cid,
            "timestamp": ts,
            "kind": kind,
            "source": source,
            "folder": folder,
            "cams": {c: str(p) for c, p in sorted(cams.items(), key=lambda kv: ingest.CAMS.index(kv[0]))},
            "event": event,
            "seconds": info.get("seconds"),
            "mph": info.get("mph"),
            "mph_1hz": info.get("mph_1hz"),
            "route": info.get("route") or [],
            "_new": new,
        }

    # ---- queries ----

    def clip(self, key: str) -> dict | None:
        with self.lock:
            return self.clips.get(key)

    def all_clips(self) -> list[dict]:
        with self.lock:
            return sorted(self.clips.values(), key=lambda c: c["timestamp"])

    def drives(self) -> list[dict]:
        return group_drives(self.all_clips())

    def drive(self, drive_id: str) -> dict | None:
        return next((d for d in self.drives() if d["id"] == drive_id), None)

    # ---- thumbnails ----

    def thumb(self, key: str, cam: str = "front", t: float | None = None) -> Path | None:
        clip = self.clip(key)
        if not clip or cam not in clip["cams"]:
            return None
        src = Path(clip["cams"][cam])
        secs = clip.get("seconds") or 2
        t = min(max(0.0, t if t is not None else min(2.0, secs / 2)), max(0.0, secs - 0.2))
        st = src.stat()
        h = hashlib.sha1(f"{src}|{st.st_size}|{int(st.st_mtime)}|{t:.1f}".encode()).hexdigest()[:16]
        out = self.thumbs / f"{h}.jpg"
        if out.exists():
            return out
        self.thumbs.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.jpg")
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(src)]
            + ["-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "5", str(tmp)],
            capture_output=True,
            timeout=60,
        )
        if r.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(out)
        return out


def _walk(root: Path, limit: int | None = None):
    n = 0
    for f in root.rglob("*.mp4"):
        if f.name.startswith("._"):
            continue
        yield f
        n += 1
        if limit and n >= limit:
            return


def group_files(root: Path) -> dict[tuple[str, str], dict[str, Path]]:
    """Per-camera files -> {(folder, timestamp): {cam: path}} (same grouping as `otrv ingest`)."""
    groups: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for f in sorted(_walk(root)):
        m = ingest.CLIP_RE.match(f.name)
        if m:
            groups[(str(f.parent), m.group(1))][m.group(2).lower()] = f
    return groups


def group_drives(clips: list[dict]) -> list[dict]:
    """Consecutive clips (gap <= DRIVE_GAP) of the same family -> drives, newest first. The same
    timestamp in several places (RecentClips and a SavedClips copy, or two sources) is shown once,
    preferring the copy with more cameras."""
    best: dict[str, dict] = {}
    for c in clips:
        fam = "sentry" if c["kind"] == "SentryClips" else "drive"
        k = f"{fam}|{c['timestamp']}"
        if k not in best or len(c["cams"]) > len(best[k]["cams"]):
            best[k] = c
    runs: list[list[dict]] = []
    for c in sorted(best.values(), key=lambda c: (c["kind"] == "SentryClips", c["timestamp"])):
        if runs:
            prev = runs[-1][-1]
            same_family = (prev["kind"] == "SentryClips") == (c["kind"] == "SentryClips")
            gap = (parse_ts(c["timestamp"]) - parse_ts(prev["timestamp"])).total_seconds() - (prev["seconds"] or 60)
            if same_family and gap <= DRIVE_GAP and (prev["kind"] != "SentryClips" or prev["folder"] == c["folder"]):
                runs[-1].append(c)
                continue
        runs.append([c])
    drives = [summarize_drive(r) for r in runs]
    drives.sort(key=lambda d: d["start"], reverse=True)
    return drives


def summarize_drive(clips: list[dict]) -> dict:
    t0 = parse_ts(clips[0]["timestamp"])
    out_clips, series, route = [], [], []
    for c in clips:
        off = (parse_ts(c["timestamp"]) - t0).total_seconds()
        out_clips.append(
            {
                "key": c["key"],
                "id": c["id"],
                "timestamp": c["timestamp"],
                "kind": c["kind"],
                "offset": off,
                "seconds": c["seconds"],
                "cams": list(c["cams"]),
                "mph": c["mph"],
                "mph_1hz": c["mph_1hz"],
                "event": (c.get("event") or {}).get("reason"),
            }
        )
        if c["mph_1hz"]:
            series += [[round(off + i, 1), v] for i, v in enumerate(c["mph_1hz"])]
        route += c.get("route") or []
    last = clips[-1]
    seconds = (parse_ts(last["timestamp"]) - t0).total_seconds() + (last["seconds"] or 0)
    weights = [(c["mph"], c["seconds"] or 0) for c in clips if c["mph"] is not None]
    wsum = sum(w for _, w in weights)
    mean = round(sum(m * w for m, w in weights) / wsum, 1) if wsum else None
    pick = speed.suggest(clips, 20, top=1)
    best = None
    if pick:
        p = pick[0]
        c = next(c for c in out_clips if c["id"] == p["id"])
        best = {
            "clip": c["key"],
            "start": p["start"],
            "offset": c["offset"] + p["start"],
            "dur": 20,
            "share": p["share"],
            "mean_mph": p["mean_mph"],
        }
    cams = max(clips, key=lambda c: len(c["cams"]))
    return {
        "id": hashlib.sha1(clips[0]["key"].encode()).hexdigest()[:10],
        "start": clips[0]["timestamp"],
        "seconds": round(seconds, 1),
        "kind": clips[0]["kind"],
        "n_clips": len(clips),
        "n_cams": len(cams["cams"]),
        "mean_mph": mean,
        "max_mph": max((v for _, v in series), default=None),
        "speed": series,
        "route": route,
        "thumb_clip": cams["key"],
        "thumb_cams": [c for c in ("front", "back", "left_repeater", "right_repeater") if c in cams["cams"]],
        "best": best,
        "clips": out_clips,
    }


def card(drive: dict) -> dict:
    """The grid's lighter view of a drive (downsampled speed + route, no per-clip series)."""
    return {
        **{k: v for k, v in drive.items() if k not in ("speed", "route", "clips")},
        "speed": downsample(drive["speed"], SPEED_POINTS),
        "route": downsample(drive["route"], ROUTE_POINTS),
    }


def suggest(drive: dict, clips: list[dict], dur: int, lo: float = 10, hi: float = 40, top: int = 5) -> list[dict]:
    """speed.suggest over one drive's clips, with clip keys and drive-relative offsets."""
    keys = {c["id"]: c for c in drive["clips"]}
    out = []
    for p in speed.suggest(clips, dur, lo, hi, top):
        c = keys.get(p["id"])
        if c:
            out.append({**p, "clip": c["key"], "offset": c["offset"] + p["start"], "dur": dur})
    return out


# ---- the per-job copy -------------------------------------------------------------------------------


def stage_clip(clip: dict, dest: Path, data: Path, log=print, cancelled=lambda: False) -> Path:
    """Copy one clip's camera files (not the whole drive) into dest/ and write dest/index.json so
    `otrv --data dest prep <clip>` reads them. Prebuilt ego masks from <data>/egomask come along."""
    dest.mkdir(parents=True, exist_ok=True)
    cams = {}
    total = sum(Path(p).stat().st_size for p in clip["cams"].values())
    log(f"copying {len(clip['cams'])} camera files ({total / 1e6:.0f} MB) of clip {clip['id']} into {dest}")
    for cam, src in clip["cams"].items():
        if cancelled():
            raise RuntimeError("cancelled")
        src_p = Path(src)
        out = dest / src_p.name
        if not (out.exists() and out.stat().st_size == src_p.stat().st_size):
            tmp = out.with_suffix(".part")
            shutil.copyfile(src_p, tmp)
            tmp.replace(out)
        cams[cam] = str(out.resolve())
    ego = data / "egomask"
    if ego.is_dir():
        (dest / "egomask").mkdir(exist_ok=True)
        for m in ego.glob("*.png"):
            shutil.copy2(m, dest / "egomask" / m.name)
    entry = {
        "id": clip["id"],
        "timestamp": clip["timestamp"],
        "kind": clip["kind"],
        "seconds": clip["seconds"],
        "cams": cams,
        "event": clip.get("event"),
        "mph": clip.get("mph"),
        "mph_1hz": clip.get("mph_1hz"),
    }
    (dest / "index.json").write_text(json.dumps([entry], indent=1))
    return dest

"""Turn `otrv prep` / `otrv recon` output (COLMAP, Brush) into stage + progress events, and estimate ETAs.

The pipeline runs under a PTY so Brush draws its progress bar. Output is split on both \\r and \\n,
ANSI escapes are stripped, and each line goes through Tracker.feed().
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Callable
from pathlib import Path

STAGES = ("frames", "masks", "features", "match", "map", "undistort", "train", "export")
LABELS = {
    "frames": "Frames",
    "masks": "Masks",
    "features": "Features",
    "match": "Match",
    "map": "Map",
    "undistort": "Undistort",
    "train": "Train",
    "export": "Export",
}

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
LINE_SPLIT = re.compile(r"[\r\n]")

# COLMAP (glog lines; searched, not matched, so the "I2026... file.cc:12]" prefix doesn't matter)
RE_FEATURES = re.compile(r"Processed file \[(\d+)/(\d+)\]")
RE_MATCH = re.compile(r"Processing block \[(\d+)/(\d+)\]")
RE_MAP = re.compile(r"Registering image #\d+ \(num_reg_frames=(\d+)\)")
RE_UNDISTORT = re.compile(r"Undistorting image \[(\d+)/(\d+)\]")
# Brush progress bar: "1234/15000 Steps (12.3/s, 18m 40s remaining)"
RE_BRUSH = re.compile(r"(\d+)/(\d+)\s+Steps \(([\d.]+)/s, (.+?) remaining\)")

# Lines that mean a stage has started (printed by tsplat/prep.py and tsplat/recon.py).
STAGE_START: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\[[^\]]+\] \d+ cams"), "frames"),
    (re.compile(r"^ego mask:"), "masks"),
    (re.compile(r"^\$ colmap feature_extractor"), "features"),
    (re.compile(r"^\$ colmap (?:matches_importer|\w+_matcher)"), "match"),
    (re.compile(r"^\$ colmap (?:global_)?mapper"), "map"),
    (re.compile(r"^\$ colmap image_undistorter"), "undistort"),
    (re.compile(r"^\$ \S*brush"), "train"),
    (re.compile(r"^splat -> "), "export"),
]


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def split_lines(buf: str) -> tuple[list[str], str]:
    """Split a chunk of terminal output on \\r and \\n: (complete lines, unterminated remainder)."""
    parts = LINE_SPLIT.split(buf)
    return parts[:-1], parts[-1]


_UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1, "ms": 0.001}


def parse_duration(text: str) -> float | None:
    """Brush/indicatif durations: "18m 40s", "1h 2m", "12s", "3.5s", "00:18:40", "18:40"."""
    text = text.strip()
    if re.fullmatch(r"\d+(?::\d{1,2}){1,2}", text):
        total = 0
        for part in text.split(":"):
            total = total * 60 + int(part)
        return float(total)
    parts = re.findall(r"([\d.]+)\s*(ms|d|h|m|s)\b", text)
    if not parts:
        return None
    return sum(float(v) * _UNITS[u] for v, u in parts)


def match_progress(line: str) -> tuple[str, int, int, dict] | None:
    """A progress line -> (stage, cur, total, extra). For the mapper, total is 0 (the caller knows it)."""
    if m := RE_BRUSH.search(line):
        rate = float(m.group(3))
        return "train", int(m.group(1)), int(m.group(2)), {"rate": rate, "remaining": parse_duration(m.group(4))}
    if m := RE_FEATURES.search(line):
        return "features", int(m.group(1)), int(m.group(2)), {}
    if m := RE_MATCH.search(line):
        return "match", int(m.group(1)), int(m.group(2)), {}
    if m := RE_MAP.search(line):
        return "map", int(m.group(1)), 0, {}
    if m := RE_UNDISTORT.search(line):
        return "undistort", int(m.group(1)), int(m.group(2)), {}
    return None


def match_stage(line: str) -> str | None:
    for rx, stage in STAGE_START:
        if rx.search(line):
            return stage
    return None


class Tracker:
    """Stage + progress state for one job. Stages only move forward; per-stage progress is monotonic.

    feed(line) returns events: {"type": "stage", "stage", "status": "start"|"done", "seconds"} and
    {"type": "progress", "stage", "cur", "total", "frac", "stage_eta"}.
    """

    def __init__(
        self,
        n_images: int | None = None,
        map_total: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        start_stage: str | None = None,
    ):
        self.n_images = n_images
        self.map_total = map_total
        self.clock = clock
        self.stage: str | None = None
        self.stage_t0 = clock()
        self.frac: dict[str, float] = {}
        self.stage_eta: float | None = None
        self.seconds: dict[str, float] = {}
        self.done: set[str] = set()
        self._undistort_pass = 0
        self._undistort_last = 0
        self._map_max = 0
        if start_stage:  # resuming: everything before start_stage finished in an earlier attempt
            for s in STAGES[: STAGES.index(start_stage)]:
                self.done.add(s)
                self.frac[s] = 1.0

    @property
    def index(self) -> int:
        return STAGES.index(self.stage) if self.stage else -1

    def _close(self, stage: str, seconds: float) -> list[dict]:
        if stage in self.done:
            return []
        self.done.add(stage)
        self.frac[stage] = 1.0
        self.seconds[stage] = round(seconds, 1)
        return [{"type": "stage", "stage": stage, "status": "done", "seconds": self.seconds[stage]}]

    def begin(self, stage: str) -> list[dict]:
        """Move to `stage` (no-op unless it's ahead of the current one), closing the stages in between."""
        if stage not in STAGES or STAGES.index(stage) <= self.index:
            return []
        now = self.clock()
        events = self._close(self.stage, now - self.stage_t0) if self.stage else []
        for skipped in STAGES[self.index + 1 : STAGES.index(stage)]:
            events += self._close(skipped, 0.0)
        self.stage, self.stage_t0, self.stage_eta = stage, now, None
        self.frac.setdefault(stage, 0.0)
        events.append({"type": "stage", "stage": stage, "status": "start"})
        return events

    def finish(self, through: str) -> list[dict]:
        """Mark every stage up to and including `through` done (the phase that ends there exited 0)."""
        events = self.begin(through)
        events += self._close(through, self.clock() - self.stage_t0)
        self.stage_eta = 0.0
        return events

    def progress(self, stage: str, cur: int, total: int, rate: float | None = None, remaining=None) -> list[dict]:
        events = self.begin(stage) if stage != self.stage else []
        if stage != self.stage or total <= 0:
            return events
        frac = max(0.0, min(1.0, cur / total))
        if frac < self.frac.get(stage, 0.0):
            return events
        self.frac[stage] = frac
        elapsed = self.clock() - self.stage_t0
        if remaining is not None:
            eta = float(remaining)
        elif rate:
            eta = (total - cur) / rate
        elif frac > 0.02:
            eta = elapsed * (1 - frac) / frac
        else:
            eta = None
        self.stage_eta = None if eta is None else round(eta, 1)
        events.append(
            {
                "type": "progress",
                "stage": stage,
                "cur": cur,
                "total": total,
                "frac": round(frac, 4),
                "stage_eta": self.stage_eta,
            }
        )
        return events

    def feed(self, line: str) -> list[dict]:
        line = strip_ansi(line).strip()
        if not line:
            return []
        events: list[dict] = []
        stage = match_stage(line)
        if stage:
            events += self.begin(stage)
        hit = match_progress(line)
        if not hit:
            return events
        stage, cur, total, extra = hit
        if stage == "map":
            self._map_max = max(self._map_max, cur)
            total = self.map_total or self.n_images or 0
            if total and self._map_max > total:  # no rig: frames are images
                total = self.n_images or self._map_max
            cur = self._map_max
        elif stage == "undistort":
            # image_undistorter runs twice (images, then masks): report both passes as one bar
            if cur < self._undistort_last:
                self._undistort_pass = min(1, self._undistort_pass + 1)
            self._undistort_last = cur
            cur, total = self._undistort_pass * total + cur, total * 2
        events += self.progress(stage, cur, total, extra.get("rate"), extra.get("remaining"))
        return events

    def is_progress_line(self, line: str) -> bool:
        return match_progress(strip_ansi(line)) is not None


# ---- ETA priors --------------------------------------------------------------------------------------

# Seconds per unit when this machine has no history yet. Units: images for every stage except train
# (steps) and export (one per run). From the reference run: 240 images, 30000 steps = 73 min on an M4.
DEFAULT_RATES = {
    "frames": 0.25,
    "masks": 0.15,
    "features": 0.5,
    "match": 0.6,
    "map": 1.2,
    "undistort": 0.25,
    "train": 0.146,
    "export": 20.0,
}
SFM_SPLIT = {"features": 0.3, "match": 0.25, "map": 0.45}  # how timings.json's single "sfm" number divides


def units(stage: str, n_images: int, steps: int) -> float:
    if stage == "train":
        return steps
    if stage == "export":
        return 1
    return n_images


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def history_rates(work: Path) -> dict[str, float]:
    """Median seconds-per-unit per stage from this machine's past runs: studio stage markers
    (work/*/stages/<stage>.done) first, else recon timings.json (sfm/undistort/train) + meta/recon_args."""
    samples: dict[str, list[float]] = {s: [] for s in STAGES}
    if not work.is_dir():
        return {}
    for run in work.iterdir():
        if not run.is_dir():
            continue
        marked = set()
        for done in (run / "stages").glob("*.done"):
            d = _read(done)
            stage, secs, n = done.stem, d.get("seconds"), d.get("units")
            if stage in samples and secs and n:
                samples[stage].append(secs / n)
                marked.add(stage)
        t = _read(run / "timings.json")
        meta = _read(run / "meta.json")
        args = _read(run / "recon_args.json")
        n_img = sum((meta.get("frames") or {}).values())
        if t.get("train") and args.get("steps") and "train" not in marked:
            samples["train"].append(t["train"] / args["steps"])
        if n_img:
            if t.get("undistort") and "undistort" not in marked:
                samples["undistort"].append(t["undistort"] / n_img)
            if t.get("sfm"):
                for s, share in SFM_SPLIT.items():
                    if s not in marked:
                        samples[s].append(t["sfm"] * share / n_img)
    return {s: statistics.median(v) for s, v in samples.items() if v}


def estimate(rates: dict[str, float], stage: str, n_images: int, steps: int) -> float:
    return (rates.get(stage) or DEFAULT_RATES[stage]) * units(stage, n_images, steps)


def eta_total(
    tracker: Tracker, rates: dict[str, float], n_images: int, steps: int, include_current: bool = True
) -> float:
    """Remaining seconds: the current stage's own ETA (or its prior minus elapsed) + priors for later stages."""
    i = tracker.index
    remaining = 0.0
    if i >= 0 and include_current and tracker.frac.get(tracker.stage, 0) < 1.0:
        if tracker.stage_eta is not None:
            remaining += tracker.stage_eta
        else:
            prior = estimate(rates, tracker.stage, n_images, steps)
            remaining += max(prior * 0.1, prior - (tracker.clock() - tracker.stage_t0))
    for s in STAGES[i + 1 :]:
        if tracker.frac.get(s, 0) < 1.0:
            remaining += estimate(rates, s, n_images, steps)
    return round(remaining, 1)


def estimate_build(rates: dict[str, float], n_images: int, steps: int) -> dict[str, float]:
    """Per-stage seconds for a build that hasn't started (shown before you press Build)."""
    return {s: round(estimate(rates, s, n_images, steps), 1) for s in STAGES}

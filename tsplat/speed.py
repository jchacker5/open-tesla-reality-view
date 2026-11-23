"""Per-clip speed from SEI telemetry. Cached in index.json at ingest; used by `otrv list` and `otrv suggest`."""

from __future__ import annotations

import math
import re

from . import sei

MPH_PER_MPS = 2.23694
DEFAULT_FPS = 36.0  # used only when the clip length is unknown


def summarize(rows: list[dict], seconds: float | None) -> dict | None:
    """SEI rows (one per frame) -> {"mean_mph", "mph_1hz"}; None if there is no telemetry.
    proto3 omits zero fields, so a row without vehicle_speed_mps means 0 m/s."""
    rows = [r for r in rows if "frame_seq_no" in r]
    if len(rows) < 2:
        return None
    seq0, seq1 = rows[0]["frame_seq_no"], rows[-1]["frame_seq_no"]
    fps = (seq1 - seq0 + 1) / seconds if seconds and seq1 > seq0 else DEFAULT_FPS  # frames / clip length
    n = max(1, math.ceil((seq1 - seq0 + 1) / fps))
    bins: list[list[float]] = [[] for _ in range(n)]
    for r in rows:
        i = min(n - 1, int((r["frame_seq_no"] - seq0) / fps))
        bins[i].append(r.get("vehicle_speed_mps", 0.0) * MPH_PER_MPS)
    series: list[float] = []
    for b in bins:  # a gap (dropped SEI) repeats the previous second
        series.append(round(sum(b) / len(b), 1) if b else (series[-1] if series else 0.0))
    allv = [r.get("vehicle_speed_mps", 0.0) * MPH_PER_MPS for r in rows]
    return {"mean_mph": round(sum(allv) / len(allv), 1), "mph_1hz": series}


def clip_speed(front: str, seconds: float | None) -> dict | None:
    try:
        return summarize(sei.extract(front), seconds)
    except (OSError, RuntimeError, ValueError):
        return None


def best_window(series: list[float], dur: int, lo: float, hi: float) -> tuple[float, int, float] | None:
    """Highest share of seconds within [lo, hi] mph over any `dur`-second window: (share, start, mean)."""
    best = None
    for s in range(0, len(series) - dur + 1):
        w = series[s : s + dur]
        share = sum(lo <= v <= hi for v in w) / dur
        mean = sum(w) / dur
        key = (share, -abs(mean - (lo + hi) / 2))  # tie-break: closest to the middle of the band
        if best is None or key > best[0]:
            best = (key, s, mean)
    return (best[0][0], best[1], round(best[2], 1)) if best else None


def suggest(
    clips: list[dict], dur: int = 20, lo: float = 10, hi: float = 40, top: int = 5, min_share: float = 0.8
) -> list[dict]:
    """Rank clips by their best `dur`-second window of sustained lo-hi mph driving. Skips Sentry
    clips (recorded parked), clips with fewer than 4 cameras, and clips without SEI speed."""
    out = []
    for c in clips:
        series = c.get("mph_1hz")
        if c.get("kind") == "SentryClips" or len(c.get("cams", {})) < 4 or not series:
            continue
        w = best_window(series, dur, lo, hi)
        if w and w[0] >= min_share:
            out.append({"id": c["id"], "timestamp": c["timestamp"], "start": w[1], "share": w[0], "mean_mph": w[2]})
    out.sort(key=lambda r: (r["share"], -abs(r["mean_mph"] - (lo + hi) / 2)), reverse=True)
    return out[:top]


def run_name(clip_id: str) -> str:
    """A work-dir-safe name for a clip, e.g. 2026-11-20_07-20-00 -> drive_2026-11-20_07-20-00."""
    return "drive_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", clip_id.split("/")[-1])

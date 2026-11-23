import json

import pytest

from tsplat import ingest, sei, speed

FPS = 36


def rows_for(mph_per_second, fps=FPS, seq0=1000):
    """Synthetic SEI rows: one per frame, speed held for each second. 0 mph rows omit the field (proto3)."""
    rows = []
    for s, mph in enumerate(mph_per_second):
        for f in range(fps):
            r = {"frame_seq_no": seq0 + s * fps + f}
            if mph:
                r["vehicle_speed_mps"] = mph / speed.MPH_PER_MPS
            rows.append(r)
    return rows


def test_summarize_bins_per_second():
    s = speed.summarize(rows_for([0, 10, 20, 30]), seconds=4.0)
    assert s["mph_1hz"] == [0.0, 10.0, 20.0, 30.0]
    assert s["mean_mph"] == 15.0
    assert speed.summarize([], 60) is None
    assert speed.summarize(rows_for([5])[:1], 60) is None


def test_best_window_prefers_sustained_band():
    series = [5] * 10 + [25] * 20 + [50] * 10
    share, start, mean = speed.best_window(series, 20, 10, 40)
    assert (share, start, mean) == (1.0, 10, 25.0)
    assert speed.best_window([20] * 5, 20, 10, 40) is None  # clip shorter than the window


def clip(cid, kind="RecentClips", series=None, cams=4):
    return {
        "id": cid,
        "timestamp": cid.split("/")[-1],
        "kind": kind,
        "cams": dict.fromkeys(range(cams)),
        "mph_1hz": series,
    }


def test_suggest_ranks_and_excludes():
    clips = [
        clip("a", series=[25] * 60),  # perfect
        clip("b", series=[5] * 30 + [20] * 18 + [60] * 12),  # best window 90% in band
        clip("ev/c", kind="SentryClips", series=[25] * 60),  # parked recording: excluded
        clip("d", series=[25] * 60, cams=2),  # too few cameras
        clip("e", series=None),  # no SEI
        clip("f", series=[80] * 60),  # highway only: below min share
    ]
    picks = speed.suggest(clips, dur=20)
    assert [p["id"] for p in picks] == ["a", "b"]
    assert picks[0]["share"] == 1.0 and picks[1]["share"] == pytest.approx(0.9)
    assert speed.run_name("ev/2026-01-01_10-00-00") == "drive_2026-01-01_10-00-00"


def test_ingest_caches_speed_from_front_sei(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "duration", lambda p: 3.0)
    calls = []

    def fake_extract(path):
        calls.append(path)
        if "03-05-05" in str(path):
            return []  # clip without SEI (old firmware)
        return rows_for([10, 20, 30])

    monkeypatch.setattr(sei, "extract", fake_extract)
    rc = tmp_path / "RecentClips"
    rc.mkdir()
    for ts in ("2026-01-02_03-04-05", "2026-01-02_03-05-05"):
        for cam in ("front", "back", "left_repeater", "right_repeater"):
            (rc / f"{ts}-{cam}.mp4").write_bytes(b"x")
    out = ingest.run(None, str(tmp_path), index_only=True)
    by_id = {c["id"]: c for c in json.loads(out.read_text())}
    assert by_id["2026-01-02_03-04-05"]["mph"] == 20.0
    assert by_id["2026-01-02_03-04-05"]["mph_1hz"] == [10.0, 20.0, 30.0]
    assert by_id["2026-01-02_03-05-05"]["mph"] is None
    assert len(calls) == 2 and all(c.endswith("-front.mp4") for c in calls)

    ingest.run(None, str(tmp_path), index_only=True)  # unchanged files: cached, SEI not re-read
    assert len(calls) == 2
    (rc / "2026-01-02_03-04-05-front.mp4").write_bytes(b"changed")
    ingest.run(None, str(tmp_path), index_only=True)
    assert len(calls) == 3

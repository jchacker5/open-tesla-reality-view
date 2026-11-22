import json

import pytest

from tsplat import ingest


@pytest.fixture
def teslacam(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "duration", lambda p: 60.0)
    recent = tmp_path / "RecentClips"
    recent.mkdir()
    for ts in ("2026-01-02_03-04-05", "2026-01-02_03-05-05"):
        for cam in ingest.CAMS:
            (recent / f"{ts}-{cam}.mp4").write_bytes(b"")
    (recent / "2026-01-02_03-06-05-front.mp4").write_bytes(b"")
    (recent / "2026-01-02_03-06-05-back.MP4").write_bytes(b"")  # case-insensitive match
    (recent / "notes.mp4").write_bytes(b"")  # ignored: doesn't match the pattern

    ev = tmp_path / "SavedClips" / "2026-01-01_10-00-00"
    ev.mkdir(parents=True)
    for cam in ("front", "back", "left_repeater", "right_repeater"):
        (ev / f"2026-01-01_09-59-00-{cam}.mp4").write_bytes(b"")
    (ev / "event.json").write_text(json.dumps({"reason": "user_interaction_honk", "city": "X"}))

    bad = tmp_path / "SentryClips" / "2026-01-03_00-00-00"
    bad.mkdir(parents=True)
    (bad / "2026-01-03_00-00-00-front.mp4").write_bytes(b"")
    (bad / "event.json").write_text("{not json")
    return tmp_path


def test_build_index(teslacam):
    clips = ingest.build_index(teslacam)
    by_id = {c["id"]: c for c in clips}
    assert [c["timestamp"] for c in clips] == sorted(c["timestamp"] for c in clips)
    assert set(by_id) == {
        "2026-01-02_03-04-05",
        "2026-01-02_03-05-05",
        "2026-01-02_03-06-05",
        "2026-01-01_10-00-00/2026-01-01_09-59-00",
        "2026-01-03_00-00-00/2026-01-03_00-00-00",
    }

    full = by_id["2026-01-02_03-04-05"]
    assert full["kind"] == "RecentClips" and full["event"] is None and full["seconds"] == 60.0
    assert list(full["cams"]) == ingest.CAMS  # ordered by CAMS

    assert set(by_id["2026-01-02_03-06-05"]["cams"]) == {"front", "back"}

    saved = by_id["2026-01-01_10-00-00/2026-01-01_09-59-00"]
    assert saved["kind"] == "SavedClips"
    assert saved["event"]["reason"] == "user_interaction_honk"
    assert list(saved["cams"]) == ["front", "back", "left_repeater", "right_repeater"]

    sentry = by_id["2026-01-03_00-00-00/2026-01-03_00-00-00"]
    assert sentry["kind"] == "SentryClips" and sentry["event"] is None  # malformed event.json tolerated


def test_run_index_only_writes_index(teslacam):
    out = ingest.run(None, str(teslacam), index_only=True)
    assert len(json.loads(out.read_text())) == 5

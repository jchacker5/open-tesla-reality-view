import json
import os
import shutil
from pathlib import Path

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


def test_build_index_ignores_appledouble_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "duration", lambda p: 60.0)
    recent = tmp_path / "RecentClips"
    recent.mkdir()
    (recent / "2026-01-02_03-04-05-front.mp4").write_bytes(b"video")
    (recent / "._2026-01-02_03-04-05-front.mp4").write_bytes(b"appledouble")
    (recent / "._2026-01-02_03-05-05-back.mp4").write_bytes(b"appledouble")
    clips = ingest.build_index(tmp_path)
    assert len(clips) == 1
    assert clips[0]["cams"]["front"].endswith("/2026-01-02_03-04-05-front.mp4")


def test_index_stores_absolute_paths(teslacam, monkeypatch):
    monkeypatch.chdir(teslacam.parent)
    clips = ingest.build_index(Path(teslacam.name))  # relative root
    for c in clips:
        for p in c["cams"].values():
            assert Path(p).is_absolute() and Path(p).exists()


def test_select_files_kinds_and_since(teslacam):
    rel = {str(p) for p in ingest.select_files(teslacam)}  # default: RecentClips + SavedClips
    assert "RecentClips/2026-01-02_03-04-05-front.mp4" in rel
    assert "SavedClips/2026-01-01_10-00-00/event.json" in rel
    assert not any(p.startswith("SentryClips") for p in rel)
    assert "RecentClips/notes.mp4" not in rel

    sentry = {str(p) for p in ingest.select_files(teslacam, ("SentryClips",))}
    assert sentry == {
        "SentryClips/2026-01-03_00-00-00/2026-01-03_00-00-00-front.mp4",
        "SentryClips/2026-01-03_00-00-00/event.json",
    }

    since = {str(p) for p in ingest.select_files(teslacam, ingest.KINDS, ingest.parse_since("2026-01-02"))}
    assert not any("2026-01-01" in p for p in since)  # the SavedClips event is from Jan 1
    assert any(p.startswith("SentryClips") for p in since)
    with pytest.raises(SystemExit, match="YYYY-MM-DD"):
        ingest.parse_since("01/02/2026")


def test_select_files_takes_plain_folder_whole(tmp_path):
    (tmp_path / "2026-01-02_03-04-05-front.mp4").write_bytes(b"")
    (tmp_path / "._2026-01-02_03-04-05-front.mp4").write_bytes(b"")
    assert [str(p) for p in ingest.select_files(tmp_path)] == ["2026-01-02_03-04-05-front.mp4"]


def test_ingest_aborts_when_it_will_not_fit(teslacam, tmp_path, monkeypatch):
    (teslacam / "RecentClips" / "2026-01-02_03-04-05-front.mp4").write_bytes(b"x" * 1000)
    monkeypatch.setattr(ingest, "free_bytes", lambda p: ingest.HEADROOM + 500)
    copied = []
    monkeypatch.setattr(ingest, "copy", lambda *a: copied.append(a))
    with pytest.raises(SystemExit, match="not enough space"):
        ingest.run(str(teslacam), str(tmp_path / "dest"))
    assert not copied


def test_bytes_to_copy_skips_files_already_copied(teslacam, tmp_path):
    f = teslacam / "RecentClips" / "2026-01-02_03-04-05-front.mp4"
    f.write_bytes(b"x" * 100)
    dest = tmp_path / "dest"
    (dest / "RecentClips").mkdir(parents=True)
    shutil.copy2(f, dest / "RecentClips" / f.name)
    total, new = ingest.bytes_to_copy(teslacam, dest, [f.relative_to(teslacam)])
    assert (total, new) == (100, 0)


@pytest.mark.parametrize(
    ("version", "flag"),
    [
        ("openrsync: protocol version 29\nrsync version 2.6.9 compatible", "--progress"),
        ("rsync  version 2.6.9  protocol version 29", "--progress"),
        ("rsync  version 3.2.7  protocol version 31", "--info=progress2"),
        ("rsync  version 3.1.0  protocol version 31", "--info=progress2"),
        ("", "--progress"),
    ],
)
def test_rsync_progress_flag(version, flag):
    assert ingest.rsync_progress_flag(version) == flag


@pytest.mark.skipif(not shutil.which("rsync"), reason="needs rsync")
def test_copy_with_rsync_copies_only_selected(teslacam, tmp_path):
    dest = tmp_path / "dest"
    files = ingest.select_files(teslacam)
    ingest.copy(teslacam, dest, files)
    copied = {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}
    assert copied == {str(f) for f in files}


def test_copy_without_rsync(teslacam, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.shutil, "which", lambda name: None)
    dest = tmp_path / "dest"
    files = ingest.select_files(teslacam)
    ingest.copy(teslacam, dest, files)
    assert {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()} == {str(f) for f in files}


def test_copy_replaces_same_size_changed_file_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.shutil, "which", lambda name: None)
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    (src / "clip.mp4").write_bytes(b"new")
    (dest / "clip.mp4").write_bytes(b"old")
    os.utime(src / "clip.mp4", ns=(2_000_000_000, 2_000_000_000))
    os.utime(dest / "clip.mp4", ns=(1_000_000_000, 1_000_000_000))
    assert ingest.bytes_to_copy(src, dest, [Path("clip.mp4")]) == (3, 3)
    real_copy = shutil.copy2

    def interrupted_copy(source, target):
        Path(target).write_bytes(b"bad")
        raise OSError("interrupted copy")

    monkeypatch.setattr(ingest.shutil, "copy2", interrupted_copy)
    with pytest.raises(OSError, match="interrupted"):
        ingest.copy(src, dest, [Path("clip.mp4")])
    assert (dest / "clip.mp4").read_bytes() == b"old"
    assert list(dest.iterdir()) == [dest / "clip.mp4"]
    monkeypatch.setattr(ingest.shutil, "copy2", real_copy)
    ingest.copy(src, dest, [Path("clip.mp4")])
    assert (dest / "clip.mp4").read_bytes() == b"new"
    assert ingest.bytes_to_copy(src, dest, [Path("clip.mp4")]) == (3, 0)


def test_speed_cache_invalidates_same_size_changed_footage(tmp_path, monkeypatch):
    recent = tmp_path / "RecentClips"
    recent.mkdir()
    front = recent / "2026-01-02_03-04-05-front.mp4"
    front.write_bytes(b"old")
    monkeypatch.setattr(ingest, "duration", lambda p: 60.0)
    calls = []

    def read_speed(path, seconds):
        calls.append(Path(path).read_bytes())
        return {"mean_mph": len(calls), "mph_1hz": [len(calls)]}

    monkeypatch.setattr(ingest.speed, "clip_speed", read_speed)
    index = ingest.build_index(tmp_path)
    (tmp_path / "index.json").write_text(json.dumps(index))
    assert ingest.build_index(tmp_path)[0]["mph"] == 1
    assert len(calls) == 1
    original = front.stat().st_mtime_ns
    front.write_bytes(b"new")
    os.utime(front, ns=(original + 1_000_000_000, original + 1_000_000_000))
    assert ingest.build_index(tmp_path)[0]["mph"] == 2
    assert calls == [b"old", b"new"]

"""setup-brush without the network: a local fake release archive + .sha256 stand in for GitHub."""

import hashlib
import io
import shutil
import tarfile
import zipfile

import pytest

from tsplat import brush


def make_tar_xz(path, members):
    with tarfile.open(path, "w:xz") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            t.addfile(info, io.BytesIO(data))


def release(tmp_path, triple="aarch64-apple-darwin", tamper=False):
    """A fake release dir served by `fetch`: <asset> and <asset>.sha256 like the real ones."""
    rel = tmp_path / "release"
    rel.mkdir()
    name = brush.asset_name(triple)
    archive = rel / name
    folder = f"brush-app-{triple}"
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr(f"{folder}/brush_app.exe", b"MZ fake")
    else:
        make_tar_xz(archive, {f"{folder}/brush_app": b"#!/bin/sh\necho brush-cli 0.3.0\n", f"{folder}/README.md": b"x"})
    real = hashlib.sha256(archive.read_bytes()).hexdigest()
    (rel / f"{name}.sha256").write_text(f"{'0' * 64 if tamper else real} *{name}\n\n")
    fetched = []

    def fetch(url, out):
        assert url.startswith(brush.BASE_URL)
        fetched.append(url.rsplit("/", 1)[-1])
        shutil.copy(rel / url.rsplit("/", 1)[-1], out)

    return fetch, fetched, {name: real}  # pinned table matching this fake archive


def test_parse_sha256_formats():
    h = "ab" * 32
    assert brush.parse_sha256(f"{h} *brush.tar.xz\n\n", "brush.tar.xz") == h
    assert brush.parse_sha256(f"{h}  brush.tar.xz\n", "brush.tar.xz") == h
    assert brush.parse_sha256(f"{h}\n", "anything") == h
    with pytest.raises(ValueError):
        brush.parse_sha256(f"{h} *other.zip\n", "brush.tar.xz")


def test_verify_accepts_match_and_rejects_mismatch(tmp_path):
    f = tmp_path / "asset.tar.xz"
    f.write_bytes(b"fake brush release")
    brush.verify(f, hashlib.sha256(b"fake brush release").hexdigest().upper())
    with pytest.raises(ValueError, match="checksum mismatch"):
        brush.verify(f, hashlib.sha256(b"something else").hexdigest())


def test_setup_downloads_verifies_and_extracts(tmp_path):
    fetch, fetched, pinned = release(tmp_path)
    exe = brush.setup(tmp_path / "opt", triple="aarch64-apple-darwin", fetch=fetch, pinned=pinned)
    assert exe == tmp_path / "opt" / "brush-app-aarch64-apple-darwin" / "brush_app"
    assert exe.read_bytes().startswith(b"#!/bin/sh") and exe.stat().st_mode & 0o111
    assert fetched == ["brush-app-aarch64-apple-darwin.tar.xz.sha256", "brush-app-aarch64-apple-darwin.tar.xz"]


def test_setup_windows_zip(tmp_path):
    fetch, _, pinned = release(tmp_path, triple="x86_64-pc-windows-msvc")
    exe = brush.setup(tmp_path / "opt", triple="x86_64-pc-windows-msvc", fetch=fetch, pinned=pinned)
    assert exe.name == "brush_app.exe" and exe.read_bytes() == b"MZ fake"


def test_setup_refuses_when_release_sha256_disagrees(tmp_path):
    fetch, _, pinned = release(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="differs from the one pinned"):
        brush.setup(tmp_path / "opt", triple="aarch64-apple-darwin", fetch=fetch, pinned=pinned)
    with pytest.raises(ValueError, match="checksum mismatch"):  # nothing pinned: the .sha256 alone still fails
        brush.setup(tmp_path / "opt", triple="aarch64-apple-darwin", fetch=fetch, pinned={})
    assert not (tmp_path / "opt").exists()  # nothing extracted


def test_setup_refuses_swapped_asset_and_checksum(tmp_path):
    """Asset and .sha256 replaced together (consistent with each other): only the pinned digest catches it."""
    fetch, _, _ = release(tmp_path)
    with pytest.raises(ValueError, match="differs from the one pinned"):
        brush.setup(tmp_path / "opt", triple="aarch64-apple-darwin", fetch=fetch)  # real pinned table
    assert not (tmp_path / "opt").exists()


def test_pinned_digests_cover_every_published_asset():
    assert set(brush.PINNED_SHA256) == {brush.asset_name(t) for t in brush.TARGETS.values()}
    assert (
        brush.PINNED_SHA256["brush-app-aarch64-apple-darwin.tar.xz"]
        == "65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c"
    )
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in brush.PINNED_SHA256.values())


def test_download_shows_progress(tmp_path, monkeypatch, capsys):
    payload = b"x" * 3_000_000

    class Resp(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(brush.urllib.request, "urlopen", lambda url, timeout: Resp(payload))
    out = tmp_path / "a.tar.xz"
    brush._download(brush.BASE_URL + "a.tar.xz", out)
    assert out.read_bytes() == payload
    shown = capsys.readouterr().out
    assert "downloading a.tar.xz" in shown and "100% of 3 MB" in shown and " 34%" in shown


def test_extract_rejects_path_traversal(tmp_path):
    bad = tmp_path / "bad.tar.xz"
    make_tar_xz(bad, {"../evil": b"x"})
    with pytest.raises(ValueError, match="unsafe"):
        brush.extract(bad, tmp_path / "out")
    assert not (tmp_path / "evil").exists()


def test_setup_on_intel_mac_explains(monkeypatch, tmp_path):
    monkeypatch.setattr(brush.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(brush.platform, "machine", lambda: "x86_64")
    with pytest.raises(SystemExit, match="no Intel-Mac build"):
        brush.setup(tmp_path, fetch=lambda u, o: pytest.fail("must not download"))

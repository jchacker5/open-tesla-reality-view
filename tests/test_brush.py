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
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if tamper:
        digest = "0" * 64
    (rel / f"{name}.sha256").write_text(f"{digest} *{name}\n\n")
    fetched = []

    def fetch(url, out):
        assert url.startswith(brush.BASE_URL)
        fetched.append(url.rsplit("/", 1)[-1])
        shutil.copy(rel / url.rsplit("/", 1)[-1], out)

    return fetch, fetched


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
    fetch, fetched = release(tmp_path)
    exe = brush.setup(tmp_path / "opt", triple="aarch64-apple-darwin", fetch=fetch)
    assert exe == tmp_path / "opt" / "brush-app-aarch64-apple-darwin" / "brush_app"
    assert exe.read_bytes().startswith(b"#!/bin/sh") and exe.stat().st_mode & 0o111
    assert fetched == ["brush-app-aarch64-apple-darwin.tar.xz.sha256", "brush-app-aarch64-apple-darwin.tar.xz"]


def test_setup_windows_zip(tmp_path):
    fetch, _ = release(tmp_path, triple="x86_64-pc-windows-msvc")
    exe = brush.setup(tmp_path / "opt", triple="x86_64-pc-windows-msvc", fetch=fetch)
    assert exe.name == "brush_app.exe" and exe.read_bytes() == b"MZ fake"


def test_setup_refuses_tampered_download(tmp_path):
    fetch, _ = release(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="checksum mismatch"):
        brush.setup(tmp_path / "opt", triple="aarch64-apple-darwin", fetch=fetch)
    assert not (tmp_path / "opt").exists()  # nothing extracted


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

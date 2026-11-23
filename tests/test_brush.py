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


# ---- setup-brush --from-source and the flag adapter --------------------------------------------------


def test_from_source_without_cargo_explains_and_runs_nothing(tmp_path):
    ran = []
    with pytest.raises(SystemExit, match="rustup.rs"):
        brush.setup_from_source(tmp_path, run=lambda *a, **k: ran.append(a), which=lambda name: None)
    assert ran == [] and not (tmp_path / "src").exists()


class FakeRun:
    """Records commands; `rev-parse` answers with `head`; `cargo build` drops a fake binary."""

    def __init__(self, install_dir, head=brush.SOURCE_COMMIT):
        self.cmds, self.install_dir, self.head = [], install_dir, head

    def __call__(self, cmd, check=False, capture_output=False, text=False, cwd=None):
        self.cmds.append((cmd, cwd))
        if "init" in cmd:
            (brush.source_dir(self.install_dir) / ".git").mkdir()
        if cmd[:2] == ["cargo", "build"]:
            exe = brush.source_exe(self.install_dir)
            exe.parent.mkdir(parents=True)
            exe.write_text("")
        return type("R", (), {"stdout": self.head + "\n", "returncode": 0})()


def test_from_source_fetches_the_pinned_commit_and_builds_brush_app(tmp_path):
    run = FakeRun(tmp_path)
    exe = brush.setup_from_source(tmp_path, run=run, which=lambda name: f"/usr/bin/{name}")
    assert exe == brush.source_exe(tmp_path) and exe.name in ("brush", "brush.exe")
    flat = [" ".join(c) for c, _ in run.cmds]
    assert any(f"fetch -q --depth 1 origin {brush.SOURCE_COMMIT}" in c for c in flat)
    assert any(c.endswith("checkout -q --force --detach FETCH_HEAD") for c in flat)
    build = [(c, cwd) for c, cwd in run.cmds if c[0] == "cargo"]
    assert build == [(["cargo", "build", "--release", "-p", "brush-app"], brush.source_dir(tmp_path))]
    assert len(brush.SOURCE_COMMIT) == 40  # a full SHA, not a branch name


def test_from_source_refuses_a_different_commit(tmp_path):
    run = FakeRun(tmp_path, head="0" * 40)
    with pytest.raises(SystemExit, match="refusing to build"):
        brush.setup_from_source(tmp_path, run=run, which=lambda name: name)
    assert not any(c[0] == "cargo" for c, _ in run.cmds)


V030 = {"--total-steps", "--export-every", "--opac-loss-weight", "--max-splats", "--seed", "--eval-save-to-disk"}
MAIN = {"--total-train-iters", "--export-every", "--opac-decay", "--max-splats", "--seed", "--eval-save-to-disk"}
ARGS = ["/b", "data", "--total-steps", "100", "--opac-loss-weight", "1e-09", "--eval-save-to-disk", "--seed", "4"]


def test_adapt_args_keeps_v030_and_translates_for_main():
    assert brush.adapt_args(ARGS, V030) == (ARGS, [])
    assert brush.adapt_args(ARGS, None) == (ARGS, [])
    out, dropped = brush.adapt_args(ARGS, MAIN)
    assert out == ["/b", "data", "--total-train-iters", "100", "--eval-save-to-disk", "--seed", "4"]
    assert dropped == ["--opac-loss-weight"]


def test_cli_flags_reads_brush_help_only(tmp_path):
    exe = tmp_path / "brush"
    exe.write_text("#!/bin/sh\necho '      --total-train-iters <N>'\necho '      --max-splats <N>'\n")
    exe.chmod(0o755)
    assert brush.cli_flags(str(exe)) == {"--total-train-iters", "--max-splats"}
    exe.write_text("#!/bin/sh\necho 'usage: something --else'\n")
    assert brush.cli_flags(str(exe)) is None  # not a Brush help page: pass flags through
    assert brush.cli_flags(str(tmp_path / "missing")) is None

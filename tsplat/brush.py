"""Where Brush (the splat trainer) lives, `otrv setup-brush` to download a verified release, and
`otrv setup-brush --from-source` to build a pinned commit of Brush's main branch.

The v0.3.0 release stays the default. Brush main (1.0.0) has the degenerate-splat fix (#479: needle and
giant low-opacity splats), the Mip-Splatting 3D filter and fisheye cameras, but renamed or dropped some
training flags; `adapt_args` translates recon's v0.3.0 flags for whichever Brush is installed."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

VERSION = "0.3.0"
BASE_URL = f"https://github.com/ArthurBrussee/brush/releases/download/v{VERSION}/"
INSTALL_DIR = Path.home() / ".local" / "opt" / "brush"
# (system, normalised machine) -> Rust target triple of the published release asset
TARGETS = {
    ("Darwin", "arm64"): "aarch64-apple-darwin",
    ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("Windows", "x86_64"): "x86_64-pc-windows-msvc",
}
# SHA-256 of each v0.3.0 asset, pinned in code. setup() also checks the release's own .sha256 file and
# refuses the download if either disagrees, so a swapped asset + checksum on the release page is caught.
PINNED_SHA256 = {
    "brush-app-aarch64-apple-darwin.tar.xz": "65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c",
    "brush-app-x86_64-unknown-linux-gnu.tar.xz": "4f0f9a8785d1951c62df26aae247c02c5bba32b00f40b06df4e1c9b867399e20",
    "brush-app-x86_64-pc-windows-msvc.zip": "b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6",
}
NO_INTEL_MAC = (
    f"Brush v{VERSION} has no Intel-Mac build (only aarch64-apple-darwin, x86_64 Linux and x86_64 Windows). "
    "Use an Apple-silicon Mac, or build Brush from source (https://github.com/ArthurBrussee/brush) "
    "and point BRUSH=/path/to/brush_app at it."
)


# `otrv setup-brush --from-source`: Brush main at this commit (2026-11-22, after the native Metal backend).
SOURCE_REPO = "https://github.com/ArthurBrussee/brush"
SOURCE_COMMIT = "d43bd1e6b232c4d85112bdf7ad1c317b3973915e"
SOURCE_MIN_RUST = (1, 98)  # highest rust-version among its dependencies (cargo refuses older rustc)
# v0.3.0 flag -> its name on Brush main. Flags the installed Brush doesn't know are dropped with a warning
# (main replaced --opac-loss-weight/--scale-loss-weight with --opac-decay and its own regularisers).
RENAMED_FLAGS = {"--total-steps": "--total-train-iters"}


def _machine(machine: str) -> str:
    m = machine.lower()
    return {"amd64": "x86_64", "x86_64": "x86_64", "aarch64": "arm64", "arm64": "arm64"}.get(m, m)


def target(system: str | None = None, machine: str | None = None) -> str | None:
    """Release triple for this platform, or None if Brush publishes no build for it."""
    system = system or platform.system()
    machine = _machine(machine or platform.machine())
    return TARGETS.get((system, machine))


def asset_name(triple: str) -> str:
    return f"brush-app-{triple}.zip" if "windows" in triple else f"brush-app-{triple}.tar.xz"


def exe_name(triple: str) -> str:
    return "brush_app.exe" if "windows" in triple else "brush_app"


def default_path(install_dir: Path = INSTALL_DIR) -> Path:
    triple = target() or "aarch64-apple-darwin"
    return install_dir / f"brush-app-{triple}" / exe_name(triple)


def path() -> str:
    """The Brush binary to run: $BRUSH if set, else where `otrv setup-brush` installs it."""
    return os.environ.get("BRUSH") or str(default_path())


def parse_sha256(text: str, name: str) -> str:
    """Read the digest for `name` from a `<hex> *<name>` (or `<hex>  <name>`) checksum file."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 1 and len(parts[0]) == 64:
            return parts[0].lower()
        if len(parts) >= 2 and parts[-1].lstrip("*") == name:
            return parts[0].lower()
    raise ValueError(f"no sha256 for {name} in checksum file")


def sha256_of(file: Path) -> str:
    h = hashlib.sha256()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(file: Path, expected: str) -> None:
    got = sha256_of(file)
    if got != expected.lower():
        raise ValueError(f"checksum mismatch for {file.name}: expected {expected}, got {got}")


def _inside(root: Path, member: str) -> bool:
    return (root / member).resolve().is_relative_to(root.resolve())


def extract(archive: Path, dest: Path) -> None:
    """Unpack a .tar.xz or .zip into dest, refusing entries that would land outside it."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            bad = [n for n in z.namelist() if not _inside(dest, n)]
            if bad:
                raise ValueError(f"unsafe paths in {archive.name}: {bad[:3]}")
            z.extractall(dest)
        return
    with tarfile.open(archive, "r:*") as t:
        bad = [m.name for m in t.getmembers() if not _inside(dest, m.name) or m.issym() or m.islnk()]
        if bad:
            raise ValueError(f"unsafe entries in {archive.name}: {bad[:3]}")
        if hasattr(tarfile, "data_filter"):
            t.extractall(dest, filter="data")
        else:  # pragma: no cover - Python < 3.11.4
            t.extractall(dest)


def _download(url: str, out: Path, chunk: int = 1 << 20) -> None:
    name = url.rsplit("/", 1)[-1]
    with urllib.request.urlopen(url, timeout=60) as r, open(out, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while block := r.read(chunk):
            f.write(block)
            done += len(block)
            pct = f" {done * 100 // total:3d}% of {total / 1e6:.0f} MB" if total else ""
            print(f"\rdownloading {name}: {done / 1e6:.1f} MB{pct}", end="", flush=True)
    print()


def check_digest(file: Path, name: str, published: str, pinned: dict[str, str]) -> None:
    """The file must match the release's .sha256 AND the digest pinned in code (when one is pinned)."""
    expected = pinned.get(name)
    if expected and published.lower() != expected:
        raise ValueError(
            f"the release's checksum for {name} ({published}) differs from the one pinned in otrv ({expected}); "
            "refusing to install"
        )
    verify(file, published)
    if expected:
        verify(file, expected)


def setup(
    install_dir: Path = INSTALL_DIR,
    triple: str | None = None,
    fetch=_download,
    pinned: dict[str, str] = PINNED_SHA256,
) -> Path:
    """Download the release asset + its .sha256, verify, extract to install_dir; return the binary path."""
    triple = triple or target()
    if triple is None:
        if platform.system() == "Darwin":
            raise SystemExit(NO_INTEL_MAC)
        raise SystemExit(
            f"Brush v{VERSION} has no build for {platform.system()} {platform.machine()}. "
            "Build it from source (https://github.com/ArthurBrussee/brush) and set BRUSH=/path/to/brush_app."
        )
    name = asset_name(triple)
    with tempfile.TemporaryDirectory() as tmp:
        archive, sumfile = Path(tmp) / name, Path(tmp) / f"{name}.sha256"
        fetch(BASE_URL + name + ".sha256", sumfile)
        fetch(BASE_URL + name, archive)
        check_digest(archive, name, parse_sha256(sumfile.read_text(), name), pinned)
        print(f"sha256 ok (release .sha256{' + pinned' if name in pinned else ''}): {name}")
        extract(archive, install_dir)
    exe = install_dir / f"brush-app-{triple}" / exe_name(triple)
    if not exe.exists():
        found = next(install_dir.rglob(exe_name(triple)), None)
        if found is None:
            raise SystemExit(f"extracted {name} but found no {exe_name(triple)} under {install_dir}")
        exe = found
    exe.chmod(exe.stat().st_mode | 0o111)
    return exe


# ---- Brush main, built from source -----------------------------------------------------------------


def source_dir(install_dir: Path = INSTALL_DIR) -> Path:
    return install_dir / "src"


def source_exe(install_dir: Path = INSTALL_DIR) -> Path:
    """brush-app's binary is `brush` on main (`brush_app` in the v0.3.0 release archives)."""
    name = "brush.exe" if platform.system() == "Windows" else "brush"
    return source_dir(install_dir) / "target" / "release" / name


NO_CARGO = (
    "building Brush from source needs Rust's cargo, which isn't on PATH. Install Rust from https://rustup.rs "
    "(`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`), open a new shell and re-run "
    "`otrv setup-brush --from-source`. Or keep the prebuilt v0.3.0 release: `otrv setup-brush`."
)


def setup_from_source(
    install_dir: Path = INSTALL_DIR,
    commit: str = SOURCE_COMMIT,
    run=subprocess.run,
    which=shutil.which,
) -> Path:
    """Fetch Brush at `commit` (a shallow fetch of that one commit) into install_dir/src and run
    `cargo build --release -p brush-app`. The first build takes 10-20 min and a few GB of disk."""
    if which("cargo") is None:
        raise SystemExit(NO_CARGO)
    if which("git") is None:
        raise SystemExit("building Brush from source needs git (macOS: `xcode-select --install`)")
    out = run(["rustc", "--version"], check=False, capture_output=True, text=True).stdout or ""
    m = re.search(r"(\d+)\.(\d+)", out)
    if m and (int(m.group(1)), int(m.group(2))) < SOURCE_MIN_RUST:
        need = ".".join(map(str, SOURCE_MIN_RUST))
        raise SystemExit(
            f"Brush {commit[:12]} needs Rust {need}+ (its burn/cubecl dependencies), but rustc is "
            f"{m.group(0)}. Run `rustup update stable`, then re-run `otrv setup-brush --from-source`."
        )
    src = source_dir(install_dir)
    src.mkdir(parents=True, exist_ok=True)
    repo = ["git", "-C", str(src)]
    if not (src / ".git").exists():
        run([*repo, "init", "-q"], check=True)
        run([*repo, "remote", "add", "origin", SOURCE_REPO], check=True)
    print(f"fetching Brush {commit[:12]} from {SOURCE_REPO}")
    run([*repo, "fetch", "-q", "--depth", "1", "origin", commit], check=True)
    run([*repo, "checkout", "-q", "--force", "--detach", "FETCH_HEAD"], check=True)
    head = run([*repo, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if head != commit:
        raise SystemExit(f"fetched Brush {head} but the pinned commit is {commit}; refusing to build")
    print("building Brush: cargo build --release -p brush-app (the first build takes 10-20 min)")
    run(["cargo", "build", "--release", "-p", "brush-app"], check=True, cwd=src)
    exe = source_exe(install_dir)
    if not exe.exists():
        raise SystemExit(f"cargo finished but {exe} is missing; check the build output above")
    return exe


# ---- one set of recon flags for either Brush -------------------------------------------------------


def version(exe: str) -> tuple[int, int, int] | None:
    """(major, minor, patch) from `<exe> --version` (v0.3.0: "brush-cli 0.3.0"), or None."""
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", (r.stdout or "") + (r.stderr or ""))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def cli_flags(exe: str) -> set[str] | None:
    """The long options the installed Brush accepts (from its --help), or None if it can't be asked."""
    try:
        r = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    flags = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", r.stdout or ""))
    if r.returncode != 0 or not flags & {"--total-steps", "--total-train-iters"}:
        return None  # not a Brush help page we understand
    return flags


def adapt_args(args: list[str], flags: set[str] | None) -> tuple[list[str], list[str]]:
    """Translate v0.3.0-style Brush arguments for a Brush that accepts `flags`: rename (RENAMED_FLAGS)
    and drop the options it doesn't know, with their value. Returns (args, dropped option names).
    flags=None (Brush couldn't be asked) passes everything through unchanged."""
    if not flags:
        return list(args), []
    out, dropped, i = [], [], 0
    while i < len(args):
        a = args[i]
        if not a.startswith("--"):
            out.append(a)
            i += 1
            continue
        has_value = i + 1 < len(args) and not args[i + 1].startswith("--")
        name = RENAMED_FLAGS.get(a, a) if a not in flags else a
        if name in flags:
            out += [name, args[i + 1]] if has_value else [name]
        else:
            dropped.append(a)
        i += 2 if has_value else 1
    return out, dropped

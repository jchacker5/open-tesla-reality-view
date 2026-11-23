"""Where Brush (the splat trainer) lives, and `otrv setup-brush` to download a verified release."""

from __future__ import annotations

import hashlib
import os
import platform
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

"""`otrv doctor`: check the machine and every external tool, printing the exact fix for each failure.
A fast subset runs automatically before ingest, prep and recon (set OTRV_SKIP_DOCTOR=1 to skip it)."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import brush

MIN_COLMAP = (4, 0)
MIN_FREE_GB = 10  # a 20 s, 4-6 camera window plus training output needs a few GB; leave headroom
MIN_RAM_GB = 16


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    warn_only: bool = False  # a failed warn-only check is reported but doesn't fail doctor/preflight


def _system() -> str:
    return platform.system()


def _install(pkg: str) -> str:
    """Install command for a package on this OS."""
    return {
        "Darwin": f"brew install {pkg}",
        "Linux": f"sudo apt install {pkg}",
        "Windows": f"winget install {'Gyan.FFmpeg' if pkg == 'ffmpeg' else pkg}",
    }.get(_system(), f"install {pkg}")


def _colmap_fix() -> str:
    return {
        "Darwin": "brew install colmap  (or `brew upgrade colmap` if an older one is installed)",
        "Linux": "conda install -c conda-forge 'colmap>=4'  (distro packages are 3.x; or build: "
        "https://colmap.github.io/install.html)",
        "Windows": "download COLMAP 4.x from https://github.com/colmap/colmap/releases and add it to PATH",
    }.get(_system(), "install COLMAP 4.0+ (https://colmap.github.io/install.html)")


def _output(cmd: list[str], timeout: float = 20) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return (r.stdout or "") + (r.stderr or "")


def check_platform() -> Check:
    system, machine = _system(), platform.machine()
    desc = f"{system} {machine}"
    if brush.target(system, machine) is None:
        fix = brush.NO_INTEL_MAC if system == "Darwin" else "use an Apple-silicon Mac (Linux/Windows x86_64 untested)"
        return Check("platform", False, f"{desc}: no Brush build for this platform", fix)
    if system != "Darwin":
        return Check("platform", True, f"{desc} (untested: the project is developed on Apple silicon)")
    return Check("platform", True, desc)


def check_tool(name: str, pkg: str | None = None) -> Check:
    path = shutil.which(name)
    if path:
        return Check(name, True, path)
    return Check(name, False, "not found on PATH", _install(pkg or name))


def colmap_version(text: str) -> tuple[int, int] | None:
    m = re.search(r"COLMAP\s+(\d+)\.(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def check_colmap() -> Check:
    exe = shutil.which("colmap")
    if not exe:
        return Check("colmap", False, "not found on PATH", _colmap_fix())
    out = _output([exe, "help"])
    if out is None:
        return Check("colmap", False, f"{exe} failed to run", _colmap_fix())
    ver = colmap_version(out)
    if ver is None or ver < MIN_COLMAP:
        found = ".".join(map(str, ver)) if ver else "unknown version"
        return Check("colmap", False, f"{found} at {exe}; need {MIN_COLMAP[0]}.{MIN_COLMAP[1]}+", _colmap_fix())
    if "rig_configurator" not in out:
        return Check("colmap", False, f"{ver[0]}.{ver[1]} at {exe} has no rig_configurator", _colmap_fix())
    return Check("colmap", True, f"{ver[0]}.{ver[1]} at {exe}")


def check_brush() -> Check:
    p = Path(brush.path())
    src = "$BRUSH" if os.environ.get("BRUSH") else "default"
    fix = "otrv setup-brush  (or set BRUSH=/path/to/brush_app)"
    if not p.exists():
        return Check("brush", False, f"not found at {p} ({src})", fix)
    if not os.access(p, os.X_OK):
        return Check("brush", False, f"{p} is not executable", f"chmod +x '{p}'")
    out = _output([str(p), "--version"])
    if out is None:
        quarantine = f"xattr -d com.apple.quarantine '{p}'  (then re-run otrv doctor)"
        return Check("brush", False, f"{p} failed to run", quarantine if _system() == "Darwin" else fix)
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
    if not m or (int(m.group(1)), int(m.group(2))) != (0, 3):
        return Check("brush", False, f"{p} reports {out.strip()[:40]!r}; need 0.3.x", fix)
    return Check("brush", True, f"{m.group(0)} at {p}")


def check_disk(path: str | Path = ".", need_gb: float = MIN_FREE_GB) -> Check:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    free = shutil.disk_usage(p).free / 2**30
    if free < need_gb:
        return Check(
            "disk",
            False,
            f"{free:.1f} GB free at {p.resolve()}; need {need_gb:.0f} GB+",
            "free up space, or run from a bigger disk (use --data for footage)",
        )
    return Check("disk", True, f"{free:.0f} GB free at {p.resolve()}")


def ram_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (AttributeError, ValueError, OSError):
        return None


def check_ram(min_gb: float = MIN_RAM_GB) -> Check:
    gb = ram_gb()
    if gb is None:
        return Check("ram", True, "unknown", warn_only=True)
    if gb < min_gb - 0.5:
        return Check(
            "ram",
            False,
            f"{gb:.0f} GB; {min_gb} GB+ recommended",
            "use fewer --steps (e.g. 7000), a lower --width (960), or a shorter --dur",
            warn_only=True,
        )
    return Check("ram", True, f"{gb:.0f} GB")


CHECKS = {
    "platform": check_platform,
    "ffmpeg": lambda: check_tool("ffmpeg"),
    "ffprobe": lambda: check_tool("ffprobe", "ffmpeg"),
    "colmap": check_colmap,
    "brush": check_brush,
    "disk": check_disk,
    "ram": check_ram,
}
# What each pipeline step needs, checked before it starts.
PREFLIGHT = {
    "ingest": ["ffprobe"],
    "egomask": ["ffmpeg"],
    "prep": ["ffmpeg", "ffprobe"],
    "recon": ["colmap", "brush"],
}


def run_checks(names: list[str] | None = None) -> list[Check]:
    return [CHECKS[n]() for n in (names or list(CHECKS))]


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        tag = "ok  " if c.ok else ("warn" if c.warn_only else "FAIL")
        lines.append(f"  {tag}  {c.name:<9} {c.detail}")
        if not c.ok and c.fix:
            lines.append(f"        fix: {c.fix}")
    return "\n".join(lines)


def doctor() -> int:
    checks = run_checks()
    print(format_checks(checks))
    bad = [c for c in checks if not c.ok and not c.warn_only]
    print(f"\n{len(bad)} problem(s) to fix." if bad else "\nall good.")
    return 1 if bad else 0


def preflight(step: str, skip: tuple[str, ...] = ()) -> None:
    """Fail early, with the fix, if a tool this step needs is missing."""
    if os.environ.get("OTRV_SKIP_DOCTOR"):
        return
    checks = run_checks([n for n in PREFLIGHT.get(step, []) if n not in skip])
    bad = [c for c in checks if not c.ok and not c.warn_only]
    if bad:
        raise SystemExit(f"`otrv {step}` can't start:\n{format_checks(bad)}\n(run `otrv doctor` for a full check)")

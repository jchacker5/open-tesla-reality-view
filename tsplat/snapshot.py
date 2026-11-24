"""`otrv snapshot`: render a frame of a reconstruction on the CPU with the C++ renderer (cpp/src/splat.cpp).

No browser or GPU, so it works over SSH and in CI. The output matches Brush's own renders of the same
view to within rounding. Cameras use the viewer's framing (viewer/index.html CHASE / DRIVER / DRONE),
placed on the car's path at time `t`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# metres: back from the car, up from the road, look-at point ahead and up, vertical fov (viewer/index.html)
FOLLOW = {"chase": (7.2, 2.7, 4.5, 0.6, 45.0), "drone": (11.0, 8.0, 2.5, 0.8, 48.0)}
DRIVER_FOV = 60.0
VIEWS = ("chase", "driver", "drone")


def pose_at(path: dict, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Front-camera position and unit heading at time t, linear between path samples (clamped to the ends)."""
    frames = path["frames"]
    ts = np.array([f["t"] for f in frames])
    t = float(np.clip(t, ts[0], ts[-1]))
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    w = 0.0 if ts[i] == ts[i - 1] else (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    p = (1 - w) * np.array(frames[i - 1]["p"]) + w * np.array(frames[i]["p"])
    f = (1 - w) * np.array(frames[i - 1]["f"]) + w * np.array(frames[i]["f"])
    return p, f / (np.linalg.norm(f) + 1e-12)


def camera(path: dict, t: float, view: str, width: int, height: int):
    import otrv_native

    up = np.array(path["up"], dtype=float)
    up /= np.linalg.norm(up)
    u = 1.0 / path["metresPerUnit"]  # scene units per metre
    p, f = pose_at(path, t)
    if view == "driver":
        eye, target, fov = p, p + f * 10 * u, DRIVER_FOV
    else:
        back, rise, ahead, aim, fov = FOLLOW[view]
        road = p - up * path.get("camHeight", 1.4) * u
        eye = road - f * back * u + up * rise * u
        target = road + f * ahead * u + up * aim * u
    return otrv_native.Camera.look_at(tuple(eye), tuple(target), tuple(up), fov, width, height)


def run(work: Path, t: float, view: str, out: Path | None, width: int, height: int, quiet: bool = False) -> Path:
    try:
        import otrv_native
    except ImportError:
        raise SystemExit("otrv snapshot needs the C++ core: `uv sync --extra native`") from None
    from PIL import Image

    if view not in VIEWS:
        raise SystemExit(f"--cam must be one of {', '.join(VIEWS)}")
    path = json.loads((work / "viewer" / "path.json").read_text())
    splats = otrv_native.load_splats(str(work / "viewer" / "scene.ply"))
    rgb, _, stats = otrv_native.render(splats, camera(path, t, view, width, height))
    out = out or work / f"snapshot_{view}_{t:g}s.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out)
    if not quiet:
        print(
            f"snapshot: {out} ({width}x{height}, {view} at {t:g} s): {stats['visible']:,} of {len(splats):,} "
            f"splats visible, {stats['ms']:.0f} ms"
        )
    return out

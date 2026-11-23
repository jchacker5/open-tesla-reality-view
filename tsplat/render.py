"""`otrv render`: a cinematic MP4 of a reconstructed drive.

capture   serve work/<name>/viewer, drive the viewer's render API (`?render=1`, window.otrvRender) in
          Playwright Chromium; per frame t: the splat alone -> splat/NNNNN.png, then car + ribbon +
          shadow on transparent -> overlay/NNNNN.png
enhance   optional: NVIDIA Difix clean-up of the splat frames (tsplat/enhance.py; separate license)
stabilize optional: flow-guided blend with the previous enhanced frame to cut diffusion flicker
encode    alpha-composite overlay over splat, label "AI-enhanced" when enhanced, H.264 via ffmpeg

Playwright is an optional extra: `uv sync --extra render && uv run playwright install chromium`.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import enhance, serve

ASPECT_SIZES = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}
CAMS = ("chase", "driver", "orbit")
LABEL = "AI-enhanced"
READY_TIMEOUT_S = 180
API_TIMEOUT_S = 10  # how long the page may take to define window.otrvRender before we give up
NEEDS_V2 = (
    "otrv render needs viewer v2: the served viewer page has no render API (window.otrvRender, "
    "opened with ?render=1). Update viewer/index.html to viewer v2 (PR #15), or pass a v2 page"
)


def check_viewer(page: bytes) -> None:
    """Fail at once, before launching a browser, when the viewer page predates the render API."""
    if b"otrvRender" not in page:
        raise SystemExit(NEEDS_V2 + ".")


@dataclass
class Options:
    out: Path
    fps: float = 30
    size: tuple[int, int] = (1920, 1080)
    aspect: str = "16:9"
    cam: str = "chase"
    start: float = 0.0
    end: float | None = None
    enhance: str = "none"
    stabilize: bool = False
    stabilize_strength: float = 0.3
    label: bool = True
    keep_frames: bool = False
    headed: bool = False
    swiftshader: bool = False
    accept_license: bool = False


# ---- options ---------------------------------------------------------------------------------------


def resolve_size(size: str | None, aspect: str | None) -> tuple[tuple[int, int], str]:
    """--size WxH and/or --aspect W:H -> ((w, h), aspect). Either alone is enough; both must agree."""
    if size is None:
        aspect = aspect or "16:9"
        if aspect not in ASPECT_SIZES:
            raise SystemExit(f"--aspect: choose from {', '.join(ASPECT_SIZES)}, got {aspect!r}")
        return ASPECT_SIZES[aspect], aspect
    try:
        w, h = (int(v) for v in size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--size must look like 1920x1080, got {size!r}") from None
    if w < 16 or h < 16 or w % 2 or h % 2:
        raise SystemExit(f"--size {size}: width and height must be even and at least 16 (H.264 yuv420p)")
    g = math.gcd(w, h)
    own = f"{w // g}:{h // g}"
    if aspect and aspect != own:
        raise SystemExit(f"--size {size} is {own}, not --aspect {aspect}; pass one or make them agree")
    return (w, h), own


def frame_times(start: float, end: float, fps: float) -> list[float]:
    """Frame timestamps on [start, end): start + i / fps."""
    if fps <= 0:
        raise SystemExit("--fps must be positive")
    if end <= start:
        raise SystemExit(f"nothing to render: --end {end:g} is not after --start {start:g}")
    n = max(1, math.floor((end - start) * fps + 1e-6))
    return [round(start + i / fps, 6) for i in range(n)]


def fname(i: int) -> str:
    return f"{i:05d}.png"


# ---- pixels ----------------------------------------------------------------------------------------


def composite(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Straight-alpha 'over': overlay RGBA (H, W, 4) onto base RGB (H, W, 3); uint8 in, uint8 out.
    Canvas toDataURL PNGs are un-premultiplied, so this is the right operator for them."""
    a = overlay[..., 3:4].astype(np.float32) / 255.0
    out = overlay[..., :3].astype(np.float32) * a + base[..., :3].astype(np.float32) * (1.0 - a)
    return out.round().clip(0, 255).astype(np.uint8)


def _font(px: int):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # Pillow < 10.1: fixed-size bitmap font
        return ImageFont.load_default()


def draw_label(img: np.ndarray, text: str = LABEL) -> np.ndarray:
    """Small honest corner tag (bottom-right): this frame was changed by a generative model."""
    h, w = img.shape[:2]
    px = max(12, round(min(w, h) / 45))
    pad, margin = px // 2, px
    im = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = _font(px)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    tw, th = x1 - x0, y1 - y0
    bx1, by1 = w - margin, h - margin
    bx0, by0 = bx1 - tw - 2 * pad, by1 - th - 2 * pad
    d.rounded_rectangle((bx0, by0, bx1, by1), radius=pad, fill=(0, 0, 0, 140))
    d.text((bx0 + pad - x0, by0 + pad - y0), text, font=font, fill=(255, 255, 255, 230))
    return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))


def load_png(data: bytes, size: tuple[int, int], mode: str) -> Image.Image:
    """Decode a captured PNG at `size`. mode RGB flattens any transparency onto black."""
    im = Image.open(io.BytesIO(data)).convert("RGBA")
    if mode == "RGB":
        im = Image.alpha_composite(Image.new("RGBA", im.size, (0, 0, 0, 255)), im).convert("RGB")
    return im if im.size == size else im.resize(size, Image.LANCZOS)


# ---- capture ---------------------------------------------------------------------------------------

_FRAME_JS = """async ({t, layer}) => {
  const R = window.otrvRender;
  await R.frame(t, layer);
  const c = R.canvas || document.querySelector('canvas');
  return c.toDataURL('image/png');
}"""

_GL_JS = """() => {
  const c = document.createElement('canvas');
  const gl = c.getContext('webgl2') || c.getContext('webgl');
  if (!gl) return null;
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
}"""


def _software(renderer: str | None) -> bool:
    r = (renderer or "").lower()
    return not r or any(s in r for s in ("swiftshader", "llvmpipe", "software"))


def chromium_args(swiftshader: bool, platform: str = sys.platform) -> list[str]:
    args = ["--hide-scrollbars", "--force-device-scale-factor=1", "--ignore-gpu-blocklist"]
    if swiftshader:  # CI / no GPU: ANGLE on SwiftShader (CPU WebGL; slow but deterministic)
        return args + ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
    if platform == "darwin":
        return args + ["--use-angle=metal", "--enable-gpu"]
    return args + ["--enable-gpu"]


def _launch(pw, headed: bool, swiftshader: bool):
    # channel="chromium" = the full browser in new-headless mode (GPU-capable), not headless-shell.
    return pw.chromium.launch(channel="chromium", headless=not headed, args=chromium_args(swiftshader))


def _webgl(browser) -> str | None:
    page = browser.new_page()
    try:
        return page.evaluate(_GL_JS)
    finally:
        page.close()


def open_browser(pw, headed: bool, swiftshader: bool):
    """Headless on the GPU if it gets one; else headed (macOS headless can land on a software GL)."""
    browser = _launch(pw, headed, swiftshader)
    renderer = _webgl(browser)
    if not swiftshader and not headed and _software(renderer):
        print(f"headless Chromium has no GPU WebGL ({renderer or 'none'}); retrying with a visible window")
        browser.close()
        browser, headed = _launch(pw, True, swiftshader), True
        renderer = _webgl(browser)
    if renderer is None:
        browser.close()
        raise SystemExit("Chromium has no WebGL here. On a machine without a GPU, pass --swiftshader.")
    print(f"WebGL: {renderer} ({'headed' if headed else 'headless'})")
    return browser


def viewer_url(port: int, aspect: str, cam: str) -> str:
    return f"http://127.0.0.1:{port}/index.html?" + urlencode({"render": 1, "aspect": aspect, "cam": cam})


def _need_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise SystemExit(
            "otrv render needs Playwright: `uv sync --extra render && uv run playwright install chromium`"
        ) from e
    return sync_playwright


def capture(viewer_dir: Path, frames: Path, opts: Options) -> list[float]:
    """Serve the viewer, render every frame's splat and overlay layers to PNG. Frames already on
    disk (same manifest) are skipped, so an interrupted capture resumes. Returns the frame times."""
    sync_playwright = _need_playwright()
    (frames / "splat").mkdir(parents=True, exist_ok=True)
    (frames / "overlay").mkdir(parents=True, exist_ok=True)
    srv = serve.make_server(viewer_dir, 0)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    errors: list[str] = []
    try:
        with sync_playwright() as pw:
            browser = open_browser(pw, opts.headed, opts.swiftshader)
            w, h = opts.size
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(viewer_url(srv.server_address[1], opts.aspect, opts.cam))
            try:
                page.wait_for_function("() => !!window.otrvRender", timeout=API_TIMEOUT_S * 1000)
            except Exception as e:
                raise SystemExit(
                    f"the viewer page did not expose window.otrvRender within {API_TIMEOUT_S} s "
                    "(otrv render needs viewer v2, which loads three.js/Spark from a CDN: check your connection)"
                    + ("; page errors: " + " | ".join(errors[-3:]) if errors else "")
                ) from e
            duration = page.evaluate(
                "async () => { await window.otrvRender.ready; return window.otrvRender.duration; }"
            )
            end = float(duration) if opts.end is None else min(opts.end, float(duration))
            times = frame_times(opts.start, end, opts.fps)
            todo = [
                i for i in range(len(times)) if not all((frames / d / fname(i)).is_file() for d in ("splat", "overlay"))
            ]
            print(f"capturing {len(todo)}/{len(times)} frames ({opts.cam}, {w}x{h}, {opts.fps:g} fps, "
                  f"t={times[0]:.2f}-{end:.2f} s of {duration:.2f})")  # fmt: skip
            screenshot = False
            t0 = time.perf_counter()
            page.set_default_timeout(READY_TIMEOUT_S * 1000)
            for k, i in enumerate(todo):
                for layer, mode in (("splat", "RGB"), ("overlay", "RGBA")):
                    url = page.evaluate(_FRAME_JS, {"t": times[i], "layer": layer})
                    data = base64.b64decode(url.split(",", 1)[1])
                    if layer == "splat" and k == 0 and _blank(data):
                        # WebGL drawing buffer already cleared (no preserveDrawingBuffer): read the
                        # composited canvas instead.
                        print("canvas readback is blank; capturing via element screenshots instead")
                        screenshot = True
                    if screenshot:
                        data = page.locator("canvas").first.screenshot(omit_background=layer == "overlay")
                    _save(load_png(data, opts.size, mode), frames / layer / fname(i))
                if k % 30 == 29 or k == len(todo) - 1:
                    rate = (k + 1) / (time.perf_counter() - t0)
                    print(f"  captured {k + 1}/{len(todo)}  ({rate:.1f} frames/s)", flush=True)
            browser.close()
    finally:
        srv.shutdown()
        srv.server_close()
    return times


def _blank(png: bytes) -> bool:
    return bool(np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))[..., 3].max() == 0)


def _save(im: Image.Image, path: Path) -> None:
    tmp = path.with_name(path.stem + ".part.png")
    im.save(tmp)
    tmp.replace(path)


# ---- encode ----------------------------------------------------------------------------------------


def encode(frames: Iterable[np.ndarray], out: Path, fps: float, size: tuple[int, int]) -> int:
    """Pipe RGB frames to ffmpeg: H.264, yuv420p, CRF 18, +faststart. Returns the frame count."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found: `brew install ffmpeg` (or your package manager)")
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]  # fmt: skip
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    try:
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
            n += 1
    finally:
        p.stdin.close()
        rc = p.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)
    return n


def final_frames(base: list[Path], overlay: list[Path], raw: list[Path], opts: Options) -> Iterable[np.ndarray]:
    """Base (enhanced or raw splat) [-> stabilised] -> overlay composited -> optional label."""
    if opts.stabilize and opts.enhance != "none":
        bases = enhance.stabilize(raw, base, opts.stabilize_strength)
    else:
        bases = (np.asarray(Image.open(p).convert("RGB")) for p in base)
    for b, o in zip(bases, overlay, strict=True):
        f = composite(b, np.asarray(Image.open(o).convert("RGBA")))
        yield draw_label(f) if opts.label and opts.enhance != "none" else f


# ---- the command -----------------------------------------------------------------------------------


def _fingerprint(viewer_dir: Path, opts: Options) -> dict:
    """What the captured frames depend on; a mismatch with the saved manifest discards them."""
    scene = viewer_dir / "scene.ply"
    st = scene.stat() if scene.exists() else None
    page = viewer_dir / "index.html"
    return {
        "size": list(opts.size), "aspect": opts.aspect, "cam": opts.cam, "fps": opts.fps,
        "start": opts.start, "end": opts.end,
        "viewer": hashlib.sha256(page.read_bytes()).hexdigest() if page.exists() else None,
        "scene": [st.st_size, int(st.st_mtime)] if st else None,
    }  # fmt: skip


def run(work: Path, viewer_html: bytes, opts: Options) -> Path:
    viewer_dir = work / "viewer"
    (viewer_dir / "index.html").write_bytes(viewer_html)
    frames = work / "render"
    images = work / "images" / "front"
    if opts.enhance != "none":  # fail before any capture or download
        enhance.require_license(opts.accept_license)
        if opts.enhance == "difix-ref" and not any(images.glob("*.jpg")):
            raise SystemExit(f"--enhance difix-ref needs the real camera frames in {images} (from `otrv prep`)")
    check_viewer(viewer_html)
    manifest = frames / "manifest.json"
    fp = _fingerprint(viewer_dir, opts)
    if frames.exists() and (not manifest.exists() or json.loads(manifest.read_text()) != fp):
        print(f"settings or scene changed since the last capture: clearing {frames}")
        shutil.rmtree(frames)
    frames.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(fp))

    timings = {}
    t0 = time.perf_counter()
    times = capture(viewer_dir, frames, opts)
    timings["capture"] = time.perf_counter() - t0
    raw = [frames / "splat" / fname(i) for i in range(len(times))]
    overlay = [frames / "overlay" / fname(i) for i in range(len(times))]

    base = raw
    if opts.enhance != "none":
        est = len(times) * enhance.SECONDS_PER_FRAME[opts.enhance]
        print(f"enhancing {len(times)} frames with {enhance.MODEL[opts.enhance]}: "
              f"up to ~{enhance.fmt_secs(est)} at M4/MPS speed (resumable: re-run the same command)")  # fmt: skip
        t0 = time.perf_counter()
        ref_for = None
        if opts.enhance == "difix-ref":
            meta = json.loads((work / "meta.json").read_text())
            ref_for = lambda i: enhance.nearest_reference(images, float(meta["fps"]), times[i])  # noqa: E731
        base = enhance.enhance_frames(raw, frames / opts.enhance, enhance.Difix(opts.enhance), ref_for)
        timings["enhance"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    n = encode(final_frames(base, overlay, raw, opts), opts.out, opts.fps, opts.size)
    timings["encode"] = time.perf_counter() - t0
    secs = ", ".join(f"{k} {v:.1f} s" for k, v in timings.items())
    print(f"render: {n} frames ({n / opts.fps:.1f} s at {opts.fps:g} fps) -> {opts.out}  [{secs}]")
    if not opts.keep_frames:
        shutil.rmtree(frames, ignore_errors=True)
        print("(frames deleted; --keep-frames keeps them for re-encoding or re-enhancing)")
    return opts.out


def env_swiftshader() -> bool:
    return os.environ.get("OTRV_RENDER_SWIFTSHADER", "").strip().lower() in ("1", "true", "yes")

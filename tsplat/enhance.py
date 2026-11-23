"""Optional AI clean-up of rendered splat frames with NVIDIA Difix, plus optical-flow stabilisation.

Difix (github.com/nv-tlabs/Difix3D) is a single-step diffusion model that removes Gaussian-splat
artifacts. It is NOT part of this MIT project: its weights (Hugging Face nvidia/difix, nvidia/difix_ref)
are under the NVIDIA License (non-commercial research and evaluation only) and build on SD-Turbo
(Stability AI Community License). Nothing NVIDIA wrote is vendored here: the pipeline class is
downloaded at runtime from a pinned commit, checked against a SHA-256, then imported; the model repos
are loaded at pinned revisions, and the remote code in them is SHA-256 checked before diffusers runs it.

Heavy dependencies (torch, diffusers, opencv) live in the `enhance` extra and are imported lazily,
so the rest of this module (license gate, reference selection, stabilisation math) works without them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

MODES = ("none", "difix", "difix-ref")
MODEL = {"difix": "nvidia/difix", "difix-ref": "nvidia/difix_ref"}
# Measured on an M4 (fp16, MPS) at 1024 px wide; used for the up-front ETA.
SECONDS_PER_FRAME = {"difix": 6.3, "difix-ref": 40.0}

# Every piece of remote code is pinned. The model repos are loaded at a fixed commit, and the Python files
# in them that diffusers imports (trust_remote_code: custom VAE/UNet classes), plus model_index.json, which
# decides which of those files get imported, are SHA-256 checked after download and before any import.
REVISION = {
    "difix": "2b0c6fb5797c26b01154dfdeb19d36e5e2eaf388",
    "difix-ref": "d4830559772a5795c9d136302c5b197d6418d3fb",
}
_VAE_PY = "a0c16e2fe489d0386b04274b25e6cec212f37264283f8ce1c042270d27250edf"
REMOTE_CODE = {
    "difix": {
        "model_index.json": "0b4316574ae102b3855c4508a13becc81b353f6455dafa6186ac37d82c8292b9",
        "vae/autoencoder_kl.py": _VAE_PY,
    },
    "difix-ref": {
        "model_index.json": "06c73e4d1c483e1c81fd0b2f8a89f021f5b31f244c217ff7d62d28f6d37cc1e1",
        "vae/autoencoder_kl.py": _VAE_PY,
        "unet/unet_2d_condition.py": "d2acb23ac4d0bd164311b6622006c7184cf5e3469086cb255cfa8b32e6a184b5",
    },
}

PIPELINE_COMMIT = "1edb0f70de735becaecb1cbe68b7d22eb3c31170"
PIPELINE_URL = f"https://raw.githubusercontent.com/nv-tlabs/Difix3D/{PIPELINE_COMMIT}/src/pipeline_difix.py"
PIPELINE_SHA256 = "2f73e2708b3f9ce560800163554f869e5e43e3a42049f67da3609f7736cbab3a"
CACHE_DIR = Path.home() / ".cache" / "otrv" / "difix"

LICENSE_ENV = "OTRV_ACCEPT_NVIDIA_LICENSE"
LICENSE_NOTICE = """\
------------------------------------------------------------------------------------------
--enhance uses NVIDIA Difix, which is NOT covered by this project's MIT license.
  * Difix weights (nvidia/difix, nvidia/difix_ref) and code: NVIDIA License,
    non-commercial research and evaluation use only.
    https://huggingface.co/nvidia/difix  https://github.com/nv-tlabs/Difix3D
  * Difix is fine-tuned from SD-Turbo: Stability AI Community License.
    https://huggingface.co/stabilityai/sd-turbo
On first use this downloads ~2.5 GB of weights from Hugging Face and NVIDIA's pipeline file
from GitHub (pinned commit, SHA-256 checked; the model repos are pinned too and their code is
SHA-256 checked before it runs). Videos made with it inherit these terms:
don't use them commercially. Frames are labelled "AI-enhanced" unless you pass --no-label.
To accept, pass --accept-nvidia-license or set OTRV_ACCEPT_NVIDIA_LICENSE=1.
------------------------------------------------------------------------------------------"""


# ---- license gate ----------------------------------------------------------------------------------


def license_accepted(flag: bool, env: dict | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(flag) or env.get(LICENSE_ENV, "").strip().lower() in ("1", "true", "yes")


def require_license(flag: bool, env: dict | None = None) -> None:
    """Print the third-party license notice; stop unless the user has explicitly accepted it."""
    if not license_accepted(flag, env):
        raise SystemExit(LICENSE_NOTICE + "\nnot accepted: nothing was downloaded.")
    print("NVIDIA Difix (non-commercial research/evaluation license; SD-Turbo base): accepted via "
          + ("--accept-nvidia-license" if flag else LICENSE_ENV))  # fmt: skip


# ---- reference frames ------------------------------------------------------------------------------


def nearest_reference(images_dir: Path, fps: float, t: float) -> Path | None:
    """The real camera frame nearest in time to viewer time t. Frames are images_dir/NNNNN.jpg,
    sampled at `fps` from the start of the prepared window (the viewer's t=0)."""
    best, best_dt = None, float("inf")
    for f in Path(images_dir).glob("*.jpg"):
        if not f.stem.isdigit():
            continue
        dt = abs(int(f.stem) / fps - t)
        if dt < best_dt or (dt == best_dt and best is not None and f.name < best.name):
            best, best_dt = f, dt
    return best


# ---- Difix -----------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_pipeline(cache_dir: Path = CACHE_DIR, url: str = PIPELINE_URL, digest: str = PIPELINE_SHA256) -> Path:
    """NVIDIA's pipeline_difix.py at the pinned commit, cached and SHA-256 verified (never vendored)."""
    dest = Path(cache_dir) / f"pipeline_difix_{digest[:12]}.py"
    if dest.is_file() and sha256(dest) == digest:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading Difix pipeline ({url.split('/')[-1]} @ {PIPELINE_COMMIT[:7]}) -> {dest.parent}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        f.write(r.read())
    got = sha256(tmp)
    if got != digest:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Difix pipeline checksum mismatch: expected {digest}, got {got}. Not importing it.")
    tmp.replace(dest)
    return dest


def verify_remote_code(mode: str, download: Callable[..., str] | None = None) -> dict[str, Path]:
    """Download the model repo's remote code files at the pinned revision (into the Hugging Face cache,
    where from_pretrained finds them) and check each SHA-256. Refuses (SystemExit) on any mismatch, so
    nothing unverified is ever imported. `download` is huggingface_hub.hf_hub_download (injectable)."""
    if download is None:
        from huggingface_hub import hf_hub_download as download
    repo, rev = MODEL[mode], REVISION[mode]
    out = {}
    for name, digest in REMOTE_CODE[mode].items():
        path = Path(download(repo_id=repo, filename=name, revision=rev))
        got = sha256(path)
        if got != digest:
            raise SystemExit(
                f"{repo}@{rev[:7]}: {name} has SHA-256 {got}, expected {digest}. "
                "Refusing to load remote code that doesn't match the pinned file."
            )
        out[name] = path
    return out


def load_pipeline(Pipeline, mode: str, dtype, download: Callable[..., str] | None = None):
    """from_pretrained at the pinned revision, after the remote code it will execute has been verified."""
    verify_remote_code(mode, download)
    # trust_remote_code: the model repos ship their own unet/vae classes (difix_ref needs them); they are
    # the files verified above, at the same revision.
    return Pipeline.from_pretrained(MODEL[mode], revision=REVISION[mode], trust_remote_code=True, torch_dtype=dtype)


def _import_pipeline(path: Path):
    spec = importlib.util.spec_from_file_location("otrv_pipeline_difix", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.DifixPipeline


def work_size(w: int, h: int, width: int = 1024) -> tuple[int, int]:
    """Difix input size: `width` wide, height scaled to keep the aspect, both multiples of 8."""
    width -= width % 8
    return width, max(8, int(round(h * width / w / 8)) * 8)


def _need(module: str):
    try:
        return __import__(module)
    except ImportError as e:
        raise SystemExit(f"--enhance needs the `enhance` extra ({module} is missing): `uv sync --extra enhance`") from e


class Difix:
    """nvidia/difix or nvidia/difix_ref on MPS, else CUDA, else CPU. fp16 on GPU, fp32 on CPU."""

    def __init__(self, mode: str):
        torch = _need("torch")
        _need("diffusers")
        self.mode = mode
        if torch.backends.mps.is_available():
            self.device, dtype = "mps", torch.float16
        elif torch.cuda.is_available():
            self.device, dtype = "cuda", torch.float16
        else:
            self.device, dtype = "cpu", torch.float32
        Pipeline = _import_pipeline(fetch_pipeline())
        t0 = time.perf_counter()
        self.pipe = load_pipeline(Pipeline, mode, dtype).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print(f"loaded {MODEL[mode]} on {self.device} ({dtype}) in {time.perf_counter() - t0:.0f} s")

    def __call__(self, img: Image.Image, ref: Image.Image | None = None) -> Image.Image:
        size = img.size
        ws = work_size(*size)
        kw = {"ref_image": ref.convert("RGB").resize(ws, Image.LANCZOS)} if ref is not None else {}
        out = self.pipe(
            "remove degradation",
            image=img.convert("RGB").resize(ws, Image.LANCZOS),
            num_inference_steps=1,
            timesteps=[199],
            guidance_scale=0.0,
            **kw,
        ).images[0]
        return out.resize(size, Image.LANCZOS)


def fmt_secs(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}h{s // 60 % 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def enhance_frames(
    src: Sequence[Path],
    out_dir: Path,
    model: Callable[[Image.Image, Image.Image | None], Image.Image],
    ref_for: Callable[[int], Path | None] | None = None,
) -> list[Path]:
    """Enhance each frame into out_dir/<same name>. Frames already there are skipped (resume);
    each output is written to a temp file and renamed, so an interrupted run never leaves half a PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = [out_dir / Path(f).name for f in src]
    todo = [i for i, o in enumerate(outs) if not o.is_file()]
    if len(todo) < len(outs):
        print(f"resuming: {len(outs) - len(todo)}/{len(outs)} frames already enhanced")
    times: list[float] = []
    for k, i in enumerate(todo):
        t0 = time.perf_counter()
        ref_path = ref_for(i) if ref_for else None
        ref = Image.open(ref_path) if ref_path else None
        img = model(Image.open(src[i]), ref)
        tmp = outs[i].with_name(outs[i].stem + ".part.png")
        img.save(tmp)
        tmp.replace(outs[i])
        times.append(time.perf_counter() - t0)
        rate = float(np.mean(times[-10:]))
        left = len(todo) - k - 1
        print(f"  enhanced {i + 1}/{len(outs)}  {times[-1]:.1f} s/frame  ETA {fmt_secs(rate * left)}", flush=True)
    return outs


# ---- stabilisation ---------------------------------------------------------------------------------


def warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Sample img at (x + flow_x, y + flow_y) for every pixel (bilinear, edges clamped).
    With flow = the flow from frame i to frame i-1, this warps frame i-1 onto frame i."""
    from scipy.ndimage import map_coordinates

    h, w = flow.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords = np.stack([yy + flow[..., 1], xx + flow[..., 0]])
    src = img.astype(np.float32)
    if src.ndim == 2:
        return map_coordinates(src, coords, order=1, mode="nearest")
    return np.stack([map_coordinates(src[..., c], coords, order=1, mode="nearest") for c in range(src.shape[2])], -1)


def fb_error(fwd: np.ndarray, bwd: np.ndarray) -> np.ndarray:
    """Forward-backward consistency error in pixels at frame i: |bwd(x) + fwd(x + bwd(x))|, with
    fwd = flow i-1 -> i and bwd = flow i -> i-1. Near zero where the flow is trustworthy;
    large at occlusions, disocclusions and flow failures."""
    return np.linalg.norm(bwd + warp(fwd, bwd), axis=-1)


def stabilize_pair(
    prev: np.ndarray,
    cur: np.ndarray,
    fwd: np.ndarray,
    bwd: np.ndarray,
    strength: float = 0.3,
    max_err: float = 1.0,
) -> np.ndarray:
    """strength * (prev warped onto cur) + (1 - strength) * cur where the flow is consistent
    (error < max_err px); elsewhere cur unchanged. uint8 in, uint8 out."""
    if strength <= 0:
        return cur
    ok = (fb_error(fwd, bwd) < max_err)[..., None]
    blend = strength * warp(prev, bwd) + (1 - strength) * cur.astype(np.float32)
    return np.where(ok, blend, cur).round().clip(0, 255).astype(np.uint8)


def farneback(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense optical flow a -> b (RGB uint8 frames) with OpenCV Farneback; (H, W, 2) as (dx, dy)."""
    cv2 = _need("cv2")
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (a, b))
    return cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 4, 21, 3, 5, 1.1, 0)


def stabilize(
    raw: Sequence[np.ndarray] | Sequence[Path],
    enhanced: Sequence[np.ndarray] | Sequence[Path],
    strength: float = 0.3,
    flow: Callable[[np.ndarray, np.ndarray], np.ndarray] = farneback,
):
    """Yield stabilised enhanced frames. Flow is measured on the raw splat renders (deterministic,
    so it tracks real camera motion, not the enhancer's flicker), then used to pull the previous
    enhanced frame onto the current one."""

    def load(x):
        return np.asarray(Image.open(x).convert("RGB")) if isinstance(x, str | Path) else x

    prev_raw = prev_enh = None
    for r, e in zip(raw, enhanced, strict=True):
        r, e = load(r), load(e)
        if prev_raw is None:
            out = e
        else:
            out = stabilize_pair(prev_enh, e, flow(prev_raw, r), flow(r, prev_raw), strength)
        prev_raw, prev_enh = r, e
        yield out

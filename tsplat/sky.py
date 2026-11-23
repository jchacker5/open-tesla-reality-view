"""Sky segmentation (prep) and transparent-sky training images (recon).

prep writes sky/<cam>/<frame>.jpg.png (255 = sky) next to masks/, plus sky.json with the mean sky
colour near the top of the frame and at the horizon; recon undistorts the masks into dense/sky/ and
copies the colours into viewer/path.json as "sky" (sRGB, 0-1), for the viewer's background gradient.

Segmenters:
  heuristic (default, no dependencies): per frame, bright and low-texture pixels in the upper half
      that connect to the top of the picture; then a running median over time per camera (sky
      doesn't flicker) and a morphological clean-up.
  model (`prep --sky-model`, `uv sync --extra quality`): UperNet-ConvNeXt-S trained on ADE20K
      (openmmlab/upernet-convnext-small, MIT licence), class "sky".

How the masks reach Brush (`recon --sky`). Brush's precedence (0.3.0, brush-dataset scene.rs
LoadImage::load): each training image has exactly ONE alpha meaning. If masks/<stem>.png exists, the
mask REPLACES the image's alpha channel and the image is "masked" (alpha_is_mask: alpha is a per-pixel
loss weight, 0 = ignored). Otherwise an image with an alpha channel is "transparent": its alpha must
be matched (match_alpha_weight) and its RGB is premultiplied, so splats there are pushed to be
see-through. There is no per-pixel mix: in one image, ignored ego pixels and transparent sky can't
coexist, and the mask file wins. Making ego pixels transparent instead would erode the road beside and
below the car, which those rays also hit.

So with --sky alpha (default) every training image gets transparent sky, by cropping the ego car out:
  - no ego/mover pixels -> RGBA .png with sky alpha = 0, no mask file;
  - ego pixels (the car's body, at the edge of the frame in every camera) -> the image is cropped to
    the largest rectangle that avoids its camera's static ego region (pixels masked in at least half
    of that camera's frames), as a new PINHOLE camera with the principal point shifted, then made RGBA
    as above. Used when the rectangle keeps >= 70% of the unmasked pixels and >= 360 px per side;
  - otherwise (movers inside the rectangle, or an ego region a rectangle can't avoid) -> keep the mask
    and add the sky to it (sky ignored).
Held-out views (`recon --eval`) are left as they are, so their renders match the uncropped ground
truth. --sky ignore masks the sky out of every image; --sky off trains on it as before. The counts
per outcome go to recon_args.json ("sky_counts") and the bench card.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

MODEL_ID = "openmmlab/upernet-convnext-small"  # MIT licence; ADE20K class 2 = sky
ADE_SKY = 2
MODES = ("alpha", "ignore", "off")
WORK_W = 320  # heuristic runs at this width


def _small(img: Image.Image) -> np.ndarray:
    w, h = img.size
    return np.asarray(img.convert("RGB").resize((WORK_W, max(1, round(h * WORK_W / w))), Image.BILINEAR), np.float32)


def candidate(rgb: np.ndarray) -> np.ndarray:
    """Per-frame sky guess on a small float RGB image (0-255): bright, smooth, in the upper half and
    connected to the top of the picture (the top 20%, so fisheye vignettes don't break it)."""
    h, w = rgb.shape[:2]
    lum = rgb.mean(2) / 255.0
    gy, gx = np.gradient(ndi.gaussian_filter(lum, 1.0))
    texture = ndi.uniform_filter(np.hypot(gx, gy), 7)
    upper = np.zeros((h, w), bool)
    upper[: h // 2] = True
    bright = lum > max(0.3, 0.75 * float(np.percentile(lum[upper], 95)))
    cand = bright & (texture < 0.02) & upper
    labels, n = ndi.label(cand)
    if n == 0:
        return cand
    top_rows = max(1, int(h * 0.2))
    touching = np.unique(labels[:top_rows][labels[:top_rows] > 0])
    sizes = ndi.sum(cand, labels, touching)
    keep = touching[sizes >= 0.002 * h * w]
    return np.isin(labels, keep)


def heuristic(frames: list[Path], keep: np.ndarray | None = None, window: int = 5) -> list[np.ndarray]:
    """Sky masks (bool, full resolution) for one camera's frames in time order. `keep` is the ego
    mask (True = scene); the car's body is never sky."""
    if not frames:
        return []
    size = Image.open(frames[0]).size
    cands = np.stack([candidate(_small(Image.open(f))) for f in frames])
    half = window // 2
    out = []
    for t in range(len(frames)):
        lo, hi = max(0, t - half), min(len(frames), t + half + 1)
        m = np.median(cands[lo:hi], axis=0) > 0.5  # temporal majority: drops one-frame glitches
        m &= ndi.binary_dilation(cands[t], iterations=2)  # ...but never far from this frame's evidence
        m = ndi.binary_closing(ndi.binary_opening(m, iterations=1), iterations=2)
        out.append(_full(m, size, keep))
    return out


def _full(m: np.ndarray, size: tuple[int, int], keep: np.ndarray | None) -> np.ndarray:
    big = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.BILINEAR)) > 127
    if keep is not None:
        big &= keep
    return big


class Model:
    """ADE20K semantic segmentation (transformers) on MPS/CUDA/CPU; sky = class 2."""

    def __init__(self, model_id: str = MODEL_ID):
        try:
            import torch
            from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        except ImportError as e:
            raise SystemExit("--sky-model needs: uv sync --extra quality") from e
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.net = UperNetForSemanticSegmentation.from_pretrained(model_id).to(self.dev).eval()

    def __call__(self, img: Image.Image) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            # 576x768: stride-32 features of 18x24, divisible by UperNet's pooling scales (1, 2, 3, 6),
            # which MPS requires; also closer to the dashcams' 4:3 than the processor's 512x512.
            x = self.proc(images=img.convert("RGB"), size={"height": 576, "width": 768}, return_tensors="pt")
            x = x["pixel_values"].to(self.dev)
            logits = self.net(pixel_values=x).logits
            logits = t.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
        return (logits.argmax(1)[0] == ADE_SKY).cpu().numpy()


def model_masks(frames: list[Path], model: Model, keep: np.ndarray | None = None) -> list[np.ndarray]:
    out = []
    for f in frames:
        m = model(Image.open(f))
        out.append(m & keep if keep is not None else m)
    return out


def colours(frames: list[Path], masks: list[np.ndarray], stride: int = 4) -> dict | None:
    """Mean sky colour (sRGB 0-1) in the top quarter of each frame's sky and at its horizon (the
    bottom quarter of the sky's rows), over every `stride`th frame. None if no sky was found."""
    top, hor = [], []
    for f, m in list(zip(frames, masks, strict=True))[::stride]:
        if m.sum() < 50:
            continue
        rgb = np.asarray(Image.open(f).convert("RGB"), np.float32) / 255
        rows = np.nonzero(m)[0]
        r0, r1 = np.percentile(rows, 25), np.percentile(rows, 75)
        rr = np.arange(m.shape[0])[:, None]
        top.append(rgb[m & (rr <= r0)])
        hor.append(rgb[m & (rr >= r1)])
    if not top:
        return None
    return {
        "top": np.concatenate(top).mean(0).round(3).tolist(),
        "horizon": np.concatenate(hor).mean(0).round(3).tolist(),
    }


def write(out: Path, cams: list[str], use_model: bool = False) -> dict | None:
    """prep: sky/<cam>/<frame>.jpg.png for every frame, and sky.json (colours, segmenter, sky share)."""
    model = Model() if use_model else None
    all_frames, all_masks, share = [], [], {}
    for cam in cams:
        frames = sorted((out / "images" / cam).glob("*.jpg"))
        if not frames:
            continue
        mpath = out / "masks" / cam / f"{frames[0].name}.png"
        keep = np.asarray(Image.open(mpath).convert("L")) > 127 if mpath.exists() else None
        masks = model_masks(frames, model, keep) if model else heuristic(frames, keep)
        sd = out / "sky" / cam
        sd.mkdir(parents=True, exist_ok=True)
        for f, m in zip(frames, masks, strict=True):
            Image.fromarray(m.astype(np.uint8) * 255).save(sd / f"{f.name}.png")
        share[cam] = round(float(np.mean([m.mean() for m in masks])), 3)
        if cam != "back":  # the fisheye back camera mostly sees road and a sliver of sky
            all_frames += frames
            all_masks += masks
    info = {"segmenter": MODEL_ID if model else "heuristic", "share": share, "colours": colours(all_frames, all_masks)}
    (out / "sky.json").write_text(json.dumps(info, indent=1))
    print(f"sky ({info['segmenter']}): " + ", ".join(f"{c} {s:.0%}" for c, s in share.items()))
    return info


CROP_MIN_KEEP = 0.7  # the crop must keep this share of an image's unmasked pixels
CROP_MIN_SIDE = 360  # Brush 0.3 fails on very small images (see recon._ensure_min_size)
CROP_BLOCK = 4  # the rectangle search runs on 4x4 blocks (a block is free only if all its pixels are)


def largest_free_rect(free: np.ndarray) -> tuple[int, int, int, int] | None:
    """(r0, r1, c0, c1) of the largest all-True axis-aligned rectangle (histogram method), or None."""
    h, w = free.shape
    heights = np.zeros(w, int)
    best, best_area = None, 0
    for r in range(h):
        heights = np.where(free[r], heights + 1, 0)
        stack: list[int] = []
        for c in range(w + 1):
            cur = heights[c] if c < w else 0
            while stack and heights[stack[-1]] >= cur:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                area = heights[top] * (c - left)
                if area > best_area:
                    best_area, best = area, (r - heights[top] + 1, r + 1, left, c)
            stack.append(c)
    return best


def crop_rect(static_masked: np.ndarray, block: int = CROP_BLOCK) -> tuple[int, int, int, int] | None:
    """Pixel rectangle (r0, r1, c0, c1) avoiding every masked pixel, searched on block x block cells."""
    h, w = static_masked.shape
    hb, wb = h // block, w // block
    if hb == 0 or wb == 0:
        return None
    blocks = static_masked[: hb * block, : wb * block].reshape(hb, block, wb, block).any(axis=(1, 3))
    r = largest_free_rect(~blocks)
    if r is None:
        return None
    return tuple(int(v) * block for v in r)


def _read_model(sparse: Path):
    cams = {}
    cam_txt = (sparse / "cameras.txt").read_text() if (sparse / "cameras.txt").exists() else ""
    for line in cam_txt.splitlines():
        p = line.split()
        if p and not line.startswith("#"):
            cams[p[0]] = p
    lines = (sparse / "images.txt").read_text().splitlines()
    header = [ln for ln in lines if ln.startswith("#")]
    body = [ln for ln in lines if not ln.startswith("#")]
    if len(body) % 2:
        body.append("")
    return cams, header, [[pose, pts] for pose, pts in zip(body[0::2], body[1::2], strict=True)]


def apply(dense: Path, mode: str, skip: set[str] | frozenset = frozenset()) -> dict[str, int]:
    """recon: turn dense/sky/<stem>.png into what Brush trains on (see the module docstring).
    Transparent images become dense/images/<cam>/<stem>.png (RGBA, cropped to avoid the car when it is
    in frame, on a new camera); the rest get the sky added to their mask. `skip`: stems left untouched
    (held-out views). Returns counts per outcome."""
    counts = {"transparent": 0, "cropped": 0, "ignored": 0, "untouched": 0, "held_out": 0}
    sparse = dense / "sparse"
    if mode == "off" or not (dense / "sky").is_dir() or not (sparse / "images.txt").exists():
        return counts
    masks_dir = dense / "images" / "masks"
    cams, header, entries = _read_model(sparse)
    by_stem = {Path(e[0].split()[9]).stem: e for e in entries if len(e[0].split()) >= 10}

    def keep_of(stem: str, shape) -> np.ndarray:
        mpath = masks_dir / f"{stem}.png"
        return np.asarray(Image.open(mpath).convert("L")) > 127 if mpath.exists() else np.ones(shape, bool)

    # per camera: the static ego region (masked in at least half its frames) and the crop that avoids it
    crops: dict[str, tuple[int, int, int, int] | None] = {}
    if mode == "alpha":
        per_cam: dict[str, list[np.ndarray]] = {}
        for stem, (pose, _) in by_stem.items():
            if (masks_dir / f"{stem}.png").exists():
                per_cam.setdefault(pose.split()[8], []).append(~keep_of(stem, (1, 1)))
        for cid, ms in per_cam.items():
            shapes = {m.shape for m in ms}
            static = np.mean(ms, axis=0) >= 0.5 if len(shapes) == 1 else None
            usable = static is not None and static.any() and cid in cams and cams[cid][1] == "PINHOLE"
            crops[cid] = crop_rect(static) if usable else None

    new_cams: dict[tuple, str] = {}
    next_cam = max(int(c) for c in cams) + 1 if cams else 1
    for stem, entry in sorted(by_stem.items()):
        pose = entry[0].split()
        name, cid = pose[9], pose[8]
        img = dense / "images" / name
        sky_png = dense / "sky" / f"{stem}.png"
        if stem in skip:
            counts["held_out"] += 1
            continue
        if not sky_png.exists() or not img.exists() or name.endswith(".png"):
            continue
        sky = np.asarray(Image.open(sky_png).convert("L")) > 127
        keep = keep_of(stem, sky.shape)
        if not sky.any():
            counts["untouched"] += 1
            continue
        rect = None
        if mode == "alpha" and not keep.all():
            rect = crops.get(cid)
            if rect is not None:
                r0, r1, c0, c1 = rect
                ok = keep[r0:r1, c0:c1].all() and min(r1 - r0, c1 - c0) >= CROP_MIN_SIDE
                rect = rect if ok and (r1 - r0) * (c1 - c0) >= CROP_MIN_KEEP * keep.sum() else None
        if mode == "alpha" and (keep.all() or rect is not None):
            rgb = np.asarray(Image.open(img).convert("RGB"))
            alpha = np.where(sky, 0, 255).astype(np.uint8)
            if rect is not None:
                r0, r1, c0, c1 = rect
                rgb, alpha = rgb[r0:r1, c0:c1], alpha[r0:r1, c0:c1]
                key = (cid, rect)
                if key not in new_cams:
                    c = cams[cid]
                    fx, fy, cx, cy = (float(v) for v in c[4:8])
                    new_cams[key] = str(next_cam)
                    cams[str(next_cam)] = [str(next_cam), "PINHOLE", str(c1 - c0), str(r1 - r0)] + [
                        repr(fx),
                        repr(fy),
                        repr(cx - c0),
                        repr(cy - r0),
                    ]
                    next_cam += 1
                pose[8] = new_cams[key]
                obs = np.array(entry[1].split(), float).reshape(-1, 3) if entry[1].strip() else np.zeros((0, 3))
                obs[:, 0] -= c0
                obs[:, 1] -= r0
                entry[1] = " ".join(f"{x:g} {y:g} {int(p)}" for x, y, p in obs)
                counts["cropped"] += 1
            Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(img.with_suffix(".png"))
            img.unlink()
            (masks_dir / f"{stem}.png").unlink(missing_ok=True)
            pose[9] = name[:-4] + ".png"
            entry[0] = " ".join(pose)
            counts["transparent"] += 1
        else:
            masks_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.where(keep & ~sky, 255, 0).astype(np.uint8)).save(masks_dir / f"{stem}.png")
            counts["ignored"] += 1
    if new_cams:
        cam_lines = [ln for ln in (sparse / "cameras.txt").read_text().splitlines() if ln.startswith("#")]
        (sparse / "cameras.txt").write_text("\n".join(cam_lines + [" ".join(c) for c in cams.values()]) + "\n")
    body = [line for e in entries for line in e]
    (sparse / "images.txt").write_text("\n".join(header + body) + "\n")
    print(
        f"sky ({mode}): {counts['transparent']} images transparent ({counts['cropped']} cropped around the car), "
        f"{counts['ignored']} with sky ignored, {counts['held_out']} held out untouched"
    )
    return counts

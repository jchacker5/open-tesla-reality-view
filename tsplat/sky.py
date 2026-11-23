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

How the masks reach Brush (`recon --sky`): Brush 0.3 (and main) gives each training image exactly
one alpha meaning. If masks/<stem>.png exists, the mask REPLACES the image's alpha and is used as a
per-pixel loss weight (alpha_is_mask: ignored pixels); otherwise an image with transparency is
"transparent": its alpha must be matched (match_alpha_weight) and its RGB is premultiplied, so the
splats there are pushed to be see-through. A mask and transparent sky in one image conflict, and the
mask wins. So with --sky alpha (default) each image gets the stronger option it can have:
  - no ego/mover pixels in its mask -> RGBA .png with sky alpha = 0, no mask file (sky transparent);
  - some ego/mover pixels -> keep the mask (ignored pixels) and add the sky to it (sky ignored).
Forcing ego pixels transparent instead would erode the road beside and below the car, which those
rays also hit. --sky ignore masks the sky out of every image; --sky off trains on it as before.
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


def apply(dense: Path, mode: str) -> dict[str, int]:
    """recon: turn dense/sky/<stem>.png into what Brush trains on (see the module docstring).
    Rewrites dense/images/<cam>/<stem>.jpg -> .png (RGBA) and images.txt for transparent images,
    or adds the sky to dense/images/masks/<stem>.png. Returns counts per outcome."""
    counts = {"transparent": 0, "ignored": 0, "untouched": 0}
    if mode == "off" or not (dense / "sky").is_dir():
        return counts
    masks_dir = dense / "images" / "masks"
    renamed = set()
    for sky_png in sorted((dense / "sky").glob("*.png")):
        stem = sky_png.stem
        cam = stem.rsplit("_", 1)[0]
        img = dense / "images" / cam / f"{stem}.jpg"
        if not img.exists():
            continue
        sky = np.asarray(Image.open(sky_png).convert("L")) > 127
        mpath = masks_dir / f"{stem}.png"
        keep = np.asarray(Image.open(mpath).convert("L")) > 127 if mpath.exists() else np.ones_like(sky)
        if not sky.any():
            counts["untouched"] += 1
        elif mode == "alpha" and keep.all():
            rgba = np.dstack([np.asarray(Image.open(img).convert("RGB")), np.where(sky, 0, 255).astype(np.uint8)])
            Image.fromarray(rgba, "RGBA").save(img.with_suffix(".png"))
            img.unlink()
            mpath.unlink(missing_ok=True)
            renamed.add(f"{cam}/{stem}.jpg")
            counts["transparent"] += 1
        else:
            masks_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.where(keep & ~sky, 255, 0).astype(np.uint8)).save(mpath)
            counts["ignored"] += 1
    if renamed:
        _rename_in_images_txt(dense / "sparse" / "images.txt", renamed)
    print(f"sky ({mode}): {counts['transparent']} images transparent, {counts['ignored']} with sky ignored")
    return counts


def _rename_in_images_txt(images_txt: Path, names: set[str]) -> None:
    out, n = [], 0
    for line in images_txt.read_text().splitlines():
        if not line.startswith("#"):
            n += 1
            if n % 2 == 1:
                head, name = line.rsplit(" ", 1)
                if name in names:
                    line = f"{head} {name[:-4]}.png"
        out.append(line)
    images_txt.write_text("\n".join(out) + "\n")

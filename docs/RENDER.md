# Rendering videos: `otrv render`

`otrv render <name>` turns a reconstructed drive into an MP4: a chase, driver, free or drone camera
flying the recorded path, with your car, its path ribbon and shadow composited on top. An
optional AI clean-up pass with NVIDIA Difix removes most of the floaters and smears a raw
Gaussian splat shows.

## Setup

**Needs viewer v2.** `otrv render` drives the viewer's render API (`window.otrvRender`, opened with
`?render=1`), which arrives with viewer v2 (PR #15). With an older `viewer/index.html` it stops at once
with "otrv render needs viewer v2", before launching a browser; if the page never defines the API (for
example the CDN is unreachable) it gives up after 10 s.

```sh
uv sync --extra render                 # Playwright (Python)
uv run playwright install chromium     # its Chromium build (~150 MB, once)
# optional, for --enhance (PyTorch + diffusers, ~2 GB; weights ~2.5 GB on first use):
uv sync --extra render --extra enhance
```

ffmpeg must be on your PATH (`brew install ffmpeg`). Like `otrv view`, rendering loads three.js
and Spark from a CDN, so it needs internet.

## Examples

```sh
# hero shot: 16:9 1080p chase cam of the whole drive
uv run otrv render drive1

# vertical 9:16 short for phones, driver's-eye view, AI-cleaned
uv run otrv render drive1 --aspect 9:16 --cam driver --enhance difix --accept-nvidia-license \
  --out work/drive1/short.mp4

# square 1:1 drone-camera loop of 6 seconds, cleaned with the reference-guided model
uv run otrv render drive1 --aspect 1:1 --cam drone --start 4 --end 10 \
  --enhance difix-ref --accept-nvidia-license --out work/drive1/loop.mp4
```

| Option | Default | What it does |
|---|---|---|
| `--out` | `work/<name>/render.mp4` | output file |
| `--fps` | 30 | output frame rate |
| `--aspect` | 16:9 | `16:9` (1920x1080), `9:16` (1080x1920) or `1:1` (1080x1080); passed to the viewer as `?aspect=` |
| `--size` | from `--aspect` | explicit `WxH` (even numbers); the aspect is derived from it |
| `--cam` | chase | `chase`, `driver`, `free` or `drone`; passed to every `otrvRender.frame(t, layer, {cam})` call |
| `--start` / `--end` | whole replay | time range in seconds of replay time |
| `--enhance` | none | `difix` or `difix-ref` (see below) |
| `--stabilize` / `--no-stabilize` | on with `--enhance` | flow-guided anti-flicker |
| `--stabilize-strength` | 0.3 | weight of the previous frame in the blend |
| `--no-label` | | omit the "AI-enhanced" label |
| `--keep-frames` | | keep all PNGs in `work/<name>/render/` (by default the raw captures are deleted after encoding and the enhanced frames are kept, so a re-run skips Difix) |
| `--clean-cache` | | delete the enhanced frames too |
| `--headed` | | show the browser window |
| `--swiftshader` | | CPU WebGL for machines without a GPU (env `OTRV_RENDER_SWIFTSHADER=1`) |

## How it works

1. **Capture.** `work/<name>/viewer/` is served locally (the same byte-range server `otrv view`
   uses, on a free port) and opened in Playwright Chromium at `index.html?render=1&aspect=…`.
   In render mode the viewer exposes
   `window.otrvRender = { ready, duration, async frame(t, layer, {cam}) }`. For every frame time `t` the
   renderer calls `frame(t, 'splat', {cam})` (the splat alone) and `frame(t, 'overlay', {cam})` (car, ribbon and
   shadow on a transparent background) and reads the canvas back as PNG into
   `render/splat/NNNNN.png` and `render/overlay/NNNNN.png`. Frames are deterministic for `t`, so
   capture is resumable: re-running the same command skips frames already on disk, and changing
   any setting, the viewer page or the splat discards them.
   On macOS Chromium runs headless on the GPU (`--use-angle=metal`); if it only gets a software
   renderer it retries with a visible window. `--swiftshader` forces CPU WebGL for CI.
2. **Enhance** (optional). Difix cleans each splat frame (overlays are added afterwards, so the
   car stays crisp and is never hallucinated).
3. **Stabilize** (on with enhance). A single-step diffusion model has no memory between frames, so
   its output flickers. For each frame after the first, optical flow (OpenCV Farneback) is measured
   between consecutive *raw* splat frames (deterministic, so the flow follows real camera motion),
   the previous enhanced frame is warped onto the current one, and the two are blended
   `0.3 · warped + 0.7 · current` wherever forward-backward flow agrees to within 1 px. Elsewhere
   (occlusions, flow failures) the current frame is used unchanged.
4. **Composite and encode.** The overlay is alpha-composited over the (enhanced) splat frame, the
   label is drawn when enhanced, and ffmpeg encodes H.264 `yuv420p`, CRF 18, `+faststart`.

## AI enhancement with NVIDIA Difix

[Difix](https://github.com/nv-tlabs/Difix3D) (NVIDIA, CVPR 2025) is a single-step image diffusion
model trained to remove the artifacts of neural reconstructions. Each frame is resized to 1024 px
wide (a multiple of 8), run once with the prompt "remove degradation" at timestep 199, and
resized back.

- `--enhance difix`: [nvidia/difix](https://huggingface.co/nvidia/difix), frame in, frame out.
- `--enhance difix-ref`: [nvidia/difix_ref](https://huggingface.co/nvidia/difix_ref), also given
  the real front-camera frame nearest in time (`work/<name>/images/front/NNNNN.jpg`, sampled at
  the prep `--fps`). It is most faithful for `--cam driver`, whose view matches the front camera;
  for the chase cam it still helps with colour and texture.

The model runs on Apple's GPU (MPS) in fp16, else CUDA fp16, else CPU fp32 (very slow).

### Timings (Apple M4, 16 GB, fp16 on MPS)

| Mode | Per frame | 20 s at 30 fps (600 frames) |
|---|---|---|
| capture (no enhance) | depends on the splat; the run prints its frames/s | |
| `difix` | ~6.3-7 s | ~65-70 min |
| `difix-ref` | ~15-40 s (measured 15-17 s at 1280x720; up to ~40 s reported) | ~2.5-7 h |
| model load | 20-30 s once per run | |

The renderer prints an estimate up front and a live ETA per frame. Enhancement is resumable:
each finished frame is saved under `work/<name>/render/<mode>/` (written atomically), so after an
interruption re-run the same command and it continues where it stopped. Tips: render at
`--fps 24` or a shorter `--start/--end` window first. Enhanced frames stay in that folder after
encoding too, so re-encoding (another `--out`, `--no-label`, stabilise settings) skips Difix and never
loads the model; `--clean-cache` deletes them.

### License notice (read before using `--enhance`)

Difix is **not** part of this MIT-licensed project and is not covered by its license:

- The Difix weights and code are under the **NVIDIA License**: **non-commercial research and
  evaluation use only**.
- Difix is fine-tuned from **SD-Turbo**, under the **Stability AI Community License**.

Videos made with `--enhance` are produced by those models; do not use them commercially. Nothing
from NVIDIA is vendored in this repository. On first use, after you accept, `otrv render`:

- downloads `src/pipeline_difix.py` from Difix3D at a pinned commit into `~/.cache/otrv/difix/`,
  checks its SHA-256 and only then imports it;
- downloads the model weights from Hugging Face into its usual cache (~2.5 GB per model), at a
  pinned commit (`enhance.REVISION`). The model repos contain custom UNet/VAE code, which diffusers
  loads with `trust_remote_code`; those `.py` files (and `model_index.json`, which names them) are
  fetched at the same commit and checked against the SHA-256s in `enhance.REMOTE_CODE` first. Any
  mismatch stops the run before anything is imported.

`otrv render --enhance …` refuses to start (and downloads nothing) until you accept, with
`--accept-nvidia-license` on the command line or `OTRV_ACCEPT_NVIDIA_LICENSE=1` in the environment.
The heavy dependencies live in the optional `enhance` extra, so a plain install never pulls them.

### The "AI-enhanced" label

Enhanced frames are partly generated: Difix can sharpen a sign it never saw clearly, invent
texture, or tidy away a real object. So enhanced renders carry a small "AI-enhanced" tag in the
bottom-right corner. `--no-label` removes it; if you share a video without the label, please say
in the caption that it was AI-enhanced. Raw renders (`--enhance none`) are pure reconstructions and
carry no label.

## Troubleshooting

- **"otrv render needs viewer v2"**: the viewer page predates the render API (`?render=1`,
  `window.otrvRender`); update to viewer v2 (PR #15).
- **"did not expose window.otrvRender within 10 s"**: the page failed to load (often the CDN for
  three.js/Spark); the message includes the page errors.
- **"Chromium has no WebGL here"**: on a machine without a GPU, pass `--swiftshader`.
- **Executable doesn't exist**: run `uv run playwright install chromium`.
- **Black or empty frames**: the renderer falls back to element screenshots when the WebGL
  canvas reads back blank; `--headed` can help on machines where headless gets no GPU.

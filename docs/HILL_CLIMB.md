# Hill-climbing reconstruction quality

The method is borrowed from
[inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill)
(MIT):
- change one lever at a time;
- measure against controls;
- log every run, failures included;
- accept a change only when it passes the gates and regresses nothing.

## The loop

```sh
uv run otrv recon drive1 --eval            # holds out every camera at fixed times for PSNR
uv run otrv bench drive1 --note "baseline"
# change ONE lever, re-run, bench with a note describing the lever
uv run otrv bench drive1 --note "fps 3 -> 4"
```

`otrv bench` prints a results card and appends a JSON line to `work/ledger.jsonl` with:
- per-camera registration;
- sparse reprojection error;
- masked held-out PSNR per camera (only for runs trained with `--eval`, from that run's final step);
- masked held-out LPIPS (needs `uv sync --extra quality`), and PSNR/LPIPS on the near-road band
  (bottom 35% of each view);
- floaters: splats inside the ego swept tube, before and after clean-up;
- haze: share of opacity in translucent splats, and what each clean-up rule removed;
- stage timings;
- the full config (every `prep` and `recon` flag), plus the machine, COLMAP and Brush versions and
  the git SHA. Brush trains with a fixed `--seed`, so two runs differ by the lever, not the dice.

Held-out PSNR/LPIPS come from Brush's eval renders, made before the post-train clean-up, so the
clean-up levers (`--prune`, `--dehaze`, `otrv clean`) show up in floaters and haze, not PSNR.

**Held-out views are chosen by time, not by count** (`tsplat/holdout.py`): every camera's frame
nearest 1.33 s, 4.00 s, 6.67 s, ... (every 8/3 s; "every 8th frame at 3 fps") is held out, and frames
within 1/3 s of a held-out frame are dropped from training. So `--fps 3` and `--fps 6` are scored on
the same moments, with the nearest training view 1/3 s away in both. `--dense-init` never sees those
views. Brush has no eval list option, so recon pads `images.txt` with placeholder entries until
Brush's `--eval-split-every 8` lands on exactly those views; `work/<name>/eval.json` lists them, and
bench refuses (records an error) if Brush rendered any other set. Ground truth is `dense_base/`, the
undistorted images before any lever edited them, and the card also shows a colour-corrected PSNR
(per-image affine colour fit), which separates geometry from exposure and white balance.

Every bench appends a line, even for a broken run (say SfM failed and there is no `dense/`): each
metric that couldn't be computed is recorded under `errors` and counts as a gate failure.

**Accept a change only if:**
- the gates pass: every camera registers at least 90% of its frames, and reprojection is at most 1.5 px;
- held-out PSNR beats the control by more than the noise. Re-run the control if unsure: training isn't deterministic, so compare medians of 2-3 runs for changes under ~0.3 dB;
- no camera's PSNR and no stage time regresses by more than 3%, unless that's the trade you want.

## Levers, roughly in order of payoff

| Lever | Where | Notes |
|---|---|---|
| Window choice | `prep --start --dur` | Steady 10-40 mph past textured scenery matters more than anything else |
| Frame rate | `prep --fps` (default 6) | More overlap; match windows scale with fps, so pairs grow ~4x from 3 to 6 fps |
| Rig prior | `recon` (default) vs `--no-rig` | The rig keeps all cameras in one model; check per-camera registration |
| Mapper | `--global-mapper` | Faster on long windows; compare registration |
| Masks | `egomask`, `prep --mask-movers` | Removes ghosts. Watch PSNR on the repeaters and back |
| Resolution | `prep --width` | Higher helps PSNR up to the source resolution; SfM gets slower |
| Training steps | `recon --steps` | Diminishing returns past ~30k |
| GPS priors | issue #2 | Should cut drift on windows over 30 s |

## Results cards

Share your best card (it's in `work/<name>/metrics.json`, and `otrv bench` prints it) in the
[Discussions](https://github.com/jchacker5/open-tesla-reality-view/discussions) "Results" category,
with chip/RAM and car (HW3/HW4). Don't attach footage.

## Pipeline v2 flags

Defaults are what `otrv prep` / `otrv recon` do with no flags. All of them land in
`recon_args.json` / `meta.json` and so on the bench card's config line.

| Flag | Default | What it does |
|---|---|---|
| `prep --fps` | 6 (was 3) | Frames per second per camera. SfM runs on every frame (see below) |
| `prep --sky-model` | off | UperNet/ADE20K sky masks instead of the brightness heuristic (`--extra quality`) |
| `recon --seed` | 42 | Brush seed, fixed so A/B runs differ by the lever |
| `recon --max-splats` | 3,000,000 with <=16 GB RAM, else 6,000,000 | Brush's splat cap (Brush's own: 10M) |
| `recon --lpips-loss-weight` | 0.05 on Brush main, 0 on v0.3.0 | Brush perceptual loss (Brush's own: 0). v0.3.0's LPIPS loss aborts on Metal (`BufferTooBig`, a 28 GB buffer) for images over ~256 px, so it is off unless you pass a value |
| `recon --opac-loss-weight` | 1e-9 | Brush opacity regulariser (v0.3.0 only; Brush main has none) |
| `recon --scale-loss-weight` | 1e-8 | Brush scale regulariser (v0.3.0 only) |
| `recon --sky` | `alpha` | `alpha`: sky transparent in training; `ignore`: masked out; `off`: trained like the rest |
| `recon --dense-init` | off | Depth Anything V2 Small points fitted to SfM, plus a road ribbon, as Brush's init (`--extra quality`) |
| `recon --color-norm` | off | LAB mean/std match of every camera to the front camera over a 4 s window |
| `recon --prune` | on | Drop splats in the ego swept tube and giant faint ones |
| `recon --dehaze` | **off** until validated on real footage | Drop near-invisible, large translucent and needle (largest/middle axis) splats, and ground-level splats far ACROSS the path |
| `--min-opacity` | 0.04 | dehaze threshold |
| `--translucent-scale` / `--translucent-opacity` | 0.8 m / 0.35 | dehaze: large and translucent |
| `--needle-ratio` / `--needle-opacity` | 25 / 0.5 | dehaze: needles |
| `--ground-fog-m` | 40 m | dehaze: ground-level splats this far across the path (perpendicular to it; road ahead of the last pose is kept) |
| `--dehaze-sky` (`--sky-fog-m`, `-saturation`, `-brightness`) | off (60 m, 0.15, 0.6) | dehaze: sky painted at a finite depth; skipped with `--sky alpha` |
| `setup-brush --from-source` | off (v0.3.0) | Build Brush main at a pinned commit; opt in with `BRUSH=~/.local/opt/brush/src/target/release/brush` |

`otrv clean <name>` re-runs prune/dehaze with other thresholds on the trained splat, no retraining.
`recon` (and `clean`) end by baking `viewer/collision.json` for drive mode (`otrv collision <name>`);
the format is documented in `tsplat/collision.py`.

### SfM cost of 6 fps

COLMAP's rig mapper runs on every frame. The alternative, SfM at 3 fps and then registering the
other frames with `image_registrator` + `bundle_adjuster`, wasn't needed: the full rig SfM at 6 fps
costs ~2.5-3x the 3 fps SfM (below), and it keeps the rig constraint (a COLMAP 4 frame is all cameras at
one instant) on every frame instead of registering images one at a time. The match windows
(same camera: 5 frames, neighbouring cameras: +-8 frames at 3 fps) are scaled by fps/3 so they cover
the same time, which makes ~4x the pairs at 6 fps.

Measured on the low-resolution smoke clip (6 cameras, 20 s, M4 32 GB, CPU SIFT):

| | 3 fps | 6 fps |
|---|---|---|
| images / pairs | 360 / 7,398 | 720 / 28,998 |
| features | 0.09 min | 0.15 min |
| matching | 0.39 min | 1.41 min |
| mapper | 0.84 min | 2.43 min |
| SfM total | ~1.3 min | 3.0-4.0 min over two runs (2.5-3x) |

Real 1280-wide footage has more features per image, so expect "a few minutes" at 3 fps to become
10-15 at 6 fps: still small next to ~75 min of training at 30k steps. If it isn't, `--fps 4` is the
middle ground.

### Brush from source

`otrv setup-brush --from-source` needs Rust's cargo (https://rustup.rs), Rust 1.98+ for the pinned
commit's burn/cubecl dependencies (`rustup update stable`; it checks first), and builds brush-app at
the pinned main commit (in `tsplat/brush.py`) into `~/.local/opt/brush/src`. Brush main renamed
`--total-steps` to `--total-train-iters` and replaced the opacity/scale loss weights with its own
regularisers; `recon` reads the installed Brush's `--help` and translates or drops (with a note) the
flags it passes, so the same command works on both. The card's header shows which Brush ran.

## Suggested A/B order (real footage)

One window, `--eval`, 30k steps, ~75 min per run on the 16 GB Mac mini. Start from a v1-equivalent
control and add one lever at a time, keeping it only if it wins (gates pass, PSNR/LPIPS better beyond
noise, no camera regresses). After the first run, use `--skip-sfm` for every training-only lever.

```sh
N=drive1; R="otrv recon $N --eval --steps 30000"
otrv prep <clip> --name $N --start <s> --dur 20 --fps 3
$R --lpips-loss-weight 0 --max-splats 10000000 --sky off --no-prune
otrv bench $N --note "v1 control (fps 3)"
```

1. `prep --fps 6` (re-prep, full SfM): the coverage lever; watch registration and SfM time.
2. `--sky alpha`: removes the sky dome haze; cheap.
3. `--max-splats 3000000`: on 16 GB this is as much about memory/time as quality.
4. `--color-norm`: per-camera PSNR on the repeaters and back.
5. `--dense-init` (`uv sync --extra quality`): road band and far geometry.
6. `--opac-loss-weight 1e-7` (1e-6 collapsed the smoke scene).
7. `--prune`, then `--dehaze` (off by default): no retraining, `otrv clean $N [--dehaze]` then bench;
   judge on floaters and haze, and by eye in `otrv view` (check the road ahead and the road surface).
8. Brush main: `otrv setup-brush --from-source`, then `BRUSH=... $R <winning flags>` vs v0.3.0
   (it drops the opacity/scale loss weights, so compare at the winning config).
9. On Brush main: `--lpips-loss-weight 0.05` (its default there) vs 0. Not on v0.3.0: it aborts.

Steps 2-6, 8 and 9 are `--skip-sfm` runs (~75 min each); step 1 adds SfM; step 7 takes seconds.


### What the quality score establishes

These scores are conditional on the camera reconstruction: COLMAP estimates camera poses and sparse
geometry before the Brush train/eval split. The split holds images out of Brush fitting, not out of
COLMAP. Do not describe these metrics as an independent end-to-end reconstruction test. The benchmark
rejects missing camera/time renders defined by prep, or renders without matching ground truth.

A blind comparison needs cameras/timestamps fixed before any scene fitting, known poses from an
independent rig/trajectory or independently estimated for scoring against frozen geometry, and unseen
drives reserved for the final test. Compare missing views and the reconstruction of stable geometry
separately from appearance metrics and from any diffusion-enhanced output.

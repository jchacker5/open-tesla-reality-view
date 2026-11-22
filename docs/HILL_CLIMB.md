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
uv run otrv recon drive1 --eval            # holds out every 8th view for PSNR
uv run otrv bench drive1 --note "baseline"
# change ONE lever, re-run, bench with a note describing the lever
uv run otrv bench drive1 --note "fps 3 -> 4"
```

`otrv bench` prints a results card and appends a JSON line to `work/ledger.jsonl` with:
- per-camera registration;
- sparse reprojection error;
- masked held-out PSNR per camera;
- stage timings;
- the full config, plus the machine, COLMAP and Brush versions and the git SHA.

**Accept a change only if:**
- the gates pass: every camera registers at least 90% of its frames, and reprojection is at most 1.5 px;
- held-out PSNR beats the control by more than the noise. Re-run the control if unsure: training isn't deterministic, so compare medians of 2-3 runs for changes under ~0.3 dB;
- no camera's PSNR and no stage time regresses by more than 3%, unless that's the trade you want.

## Levers, roughly in order of payoff

| Lever | Where | Notes |
|---|---|---|
| Window choice | `prep --start --dur` | Steady 10-40 mph past textured scenery matters more than anything else |
| Frame rate | `prep --fps` | More overlap for slow driving; costs SfM time roughly quadratically in pairs |
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

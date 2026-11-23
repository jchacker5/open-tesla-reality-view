"""Held-out views for `recon --eval`, chosen by timestamp so runs at different --fps are scored on the
same moments of the drive, with the same distance to the nearest training view.

  eval times    1.33 s, 4.00 s, 6.67 s, ... (every 8/3 s from 4/3 s): frame 4, 12, 20, ... at 3 fps,
                i.e. "every 8th frame at 3 fps", mapped to the same times at any fps. Every camera's
                frame nearest each time is held out (the whole rig frame).
  guard         training frames closer than 1/3 s to a held-out frame are dropped from training, so the
                nearest training view is >= 1/3 s away at any fps (at 3 fps nothing is dropped; at 6 fps
                the two neighbours are). Without it, 6 fps would be scored on views 1/6 s from training.

Brush has no explicit eval list: with `--eval-split-every 8` it sorts images.txt by NAME and holds out
entry i when i % 8 == 0, counting entries whose image file doesn't exist (Brush 0.3.0,
brush-dataset/src/formats/colmap.rs). So `apply` inserts placeholder entries (names like
"<previous name>~0003", with no file, which Brush skips) until every held-out image sits at a multiple
of 8 and no training image does. It writes eval.json; bench checks Brush's eval renders against it and
records any mismatch (e.g. a Brush whose split works differently) as an error.

`apply` edits dense/sparse/images.txt only for Brush; everything that reads the model afterwards
(bench registration, nvs-export) reads dense_base/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

EVAL_PERIOD_S = 8 / 3
EVAL_OFFSET_S = 4 / 3
GUARD_S = 1 / 3
BRUSH_SPLIT = 8  # recon passes --eval-split-every BRUSH_SPLIT


def eval_times(dur: float) -> list[float]:
    out, t = [], EVAL_OFFSET_S
    while t < dur - 1e-9:
        out.append(round(t, 6))
        t += EVAL_PERIOD_S
    return out


def frame_index(name: str) -> int:
    """<cam>/<cam>_00012.jpg (or .png, or a bare stem) -> 12."""
    return int(Path(name).stem.rsplit("_", 1)[-1])


@dataclass
class Split:
    eval: set[str] = field(default_factory=set)  # image stems (<cam>_<idx>)
    guard: set[str] = field(default_factory=set)
    times: list[float] = field(default_factory=list)

    @property
    def held(self) -> set[str]:
        """Every view that must not inform training: held out plus guard."""
        return self.eval | self.guard


def plan(names: list[str], fps: float, dur: float) -> Split:
    """Held-out and guard stems among the registered image names, for frames sampled at `fps`."""
    times = eval_times(dur)
    eval_idx = {round(t * fps) for t in times}
    guard_idx = {
        i + d
        for i in eval_idx
        for d in range(-int(GUARD_S * fps) - 1, int(GUARD_S * fps) + 2)
        if d and abs(d) / fps < GUARD_S - 1e-6
    }
    split = Split(times=times)
    for n in names:
        stem, idx = Path(n).stem, frame_index(n)
        if idx in eval_idx:
            split.eval.add(stem)
        elif idx in guard_idx:
            split.guard.add(stem)
    return split


def _read(images_txt: Path) -> tuple[list[str], list[tuple[str, str]]]:
    lines = images_txt.read_text().splitlines()
    header = [line for line in lines if line.startswith("#")]
    body = [line for line in lines if not line.startswith("#")]
    if len(body) % 2:
        body.append("")
    return header, list(zip(body[0::2], body[1::2], strict=True))


def layout(names: list[str], eval_stems: set[str], every: int = BRUSH_SPLIT) -> list[tuple[str, str | None]]:
    """Brush's sorted order with placeholders: [(name, real name or None for a placeholder)], such that
    position % every == 0 exactly for the held-out images (and placeholders)."""
    seq: list[tuple[str, str | None]] = []
    prev, k = "!", 0

    def pad():
        nonlocal k
        seq.append((f"{prev}~{k:04d}", None))
        k += 1

    for n in sorted(names):
        if Path(n).stem in eval_stems:
            while len(seq) % every:
                pad()
        elif len(seq) % every == 0:
            pad()
        seq.append((n, n))
        prev, k = n, 0
    return seq


def apply(work: Path, dense: Path, split: Split, every: int = BRUSH_SPLIT) -> dict:
    """Drop guard images from dense/sparse/images.txt, insert placeholders so Brush holds out exactly
    split.eval, and write work/eval.json."""
    images_txt = dense / "sparse" / "images.txt"
    header, entries = _read(images_txt)
    by_name = {pose.split()[9]: (pose, pts) for pose, pts in entries if len(pose.split()) >= 10}
    keep = [n for n in by_name if Path(n).stem not in split.guard]
    seq = layout(keep, split.eval, every)
    any_pose = next(iter(by_name.values()))[0].split()
    next_id = max(int(pose.split()[0]) for pose, _ in by_name.values()) + 1
    out = list(header)
    for name, real in seq:
        if real is None:
            out += [" ".join([str(next_id), *any_pose[1:9], name]), ""]
            next_id += 1
        else:
            out += list(by_name[real])
    images_txt.write_text("\n".join(out) + "\n")
    held = sorted(Path(n).stem for n in keep if Path(n).stem in split.eval)
    info = {
        "period_s": round(EVAL_PERIOD_S, 6),
        "offset_s": round(EVAL_OFFSET_S, 6),
        "guard_s": round(GUARD_S, 6),
        "times": split.times,
        "eval": held,
        "guard": sorted(split.guard & {Path(n).stem for n in by_name}),
        "placeholders": sum(1 for _, r in seq if r is None),
        "split_every": every,
    }
    (work / "eval.json").write_text(json.dumps(info, indent=1))
    print(
        f"held out: {len(held)} views at {len(split.times)} times, {len(info['guard'])} guard frames dropped "
        f"from training, {info['placeholders']} placeholders for Brush's split"
    )
    return info

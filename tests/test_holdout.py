"""Held-out views by timestamp, and the placeholder layout that makes Brush's `i % 8 == 0` split pick them."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tsplat import bench, holdout

CAMS = ["back", "front", "left_repeater", "right_repeater"]


def names(fps, dur=20, cams=CAMS):
    return [f"{c}/{c}_{i:05d}.jpg" for c in cams for i in range(int(fps * dur))]


def brush_eval(entries: list[str], every=8) -> set[str]:
    """Brush 0.3.0: sort images.txt names, hold out i % every == 0, skip names with no image file."""
    real = {n for n in entries if "~" not in n}
    return {Path(n).stem for i, n in enumerate(sorted(entries)) if i % every == 0 and n in real}


@pytest.mark.parametrize("fps", [3, 4, 6])
def test_same_times_at_every_fps_and_guard_keeps_training_views_a_third_of_a_second_away(fps):
    split = holdout.plan(names(fps), fps, 20)
    assert split.times == pytest.approx([4 / 3 + k * 8 / 3 for k in range(7)])
    times = sorted({holdout.frame_index(s) / fps for s in split.eval})
    assert times == pytest.approx(split.times, abs=0.5 / fps)  # the frame nearest each time
    assert len(split.eval) == 7 * len(CAMS)  # the same number of views at any fps
    train = [holdout.frame_index(n) / fps for n in names(fps) if Path(n).stem not in split.held]
    nearest = min(abs(t - e) for e in times for t in train)
    assert nearest >= 1 / 3 - 1e-6


def test_three_fps_is_every_eighth_frame_and_drops_nothing():
    split = holdout.plan(names(3), 3, 20)
    assert {holdout.frame_index(s) for s in split.eval} == {4, 12, 20, 28, 36, 44, 52}
    assert split.guard == set()
    six = holdout.plan(names(6), 6, 20)
    assert {holdout.frame_index(s) for s in six.eval} == {8, 24, 40, 56, 72, 88, 104}
    assert {holdout.frame_index(s) for s in six.guard} == {7, 9, 23, 25, 39, 41, 55, 57, 71, 73, 87, 89, 103, 105}


@pytest.mark.parametrize("fps", [3, 6])
def test_layout_makes_brush_hold_out_exactly_the_eval_views(fps):
    split = holdout.plan(names(fps), fps, 20)
    keep = [n for n in names(fps) if Path(n).stem not in split.guard]
    seq = holdout.layout(keep, split.eval)
    assert [n for n, _ in seq] == sorted(n for n, _ in seq)  # placeholders sort where they were put
    assert {r for _, r in seq if r} == set(keep)
    assert brush_eval([n for n, _ in seq]) == split.eval


def test_apply_rewrites_images_txt_and_writes_eval_json(tmp_path):
    fps = 6
    work, dense = tmp_path / "w", tmp_path / "w" / "dense"
    (dense / "sparse").mkdir(parents=True)
    all_names = names(fps, dur=6, cams=["front", "back"])
    lines = ["# header"]
    for i, n in enumerate(all_names):
        lines += [f"{i + 1} 1 0 0 0 0 0 {i} 1 {n}", f"1 2 {i + 1}"]
    (dense / "sparse" / "images.txt").write_text("\n".join(lines) + "\n")
    split = holdout.plan(all_names, fps, 6)
    info = holdout.apply(work, dense, split)
    body = [ln for ln in (dense / "sparse" / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    entries = [p.split()[9] for p in body[0::2]]
    assert brush_eval(entries) == split.eval == set(info["eval"])
    assert not {Path(n).stem for n in entries} & split.guard  # guard frames are not trained on
    ids = [int(p.split()[0]) for p in body[0::2]]
    assert len(set(ids)) == len(ids)
    real = {p.split()[9]: q for p, q in zip(body[0::2], body[1::2], strict=True) if "~" not in p}
    assert real["front/front_00000.jpg"] == "1 2 1"  # observations kept with their image
    assert json.loads((work / "eval.json").read_text())["placeholders"] == info["placeholders"] > 0


def test_bench_rejects_renders_that_are_not_the_planned_views(tmp_path):
    work = tmp_path / "run"
    (work / "dense" / "images" / "front").mkdir(parents=True)
    (work / "splat" / "eval_100").mkdir(parents=True)
    img = np.full((8, 8, 3), 100, np.uint8)
    for stem in ("front_00008", "front_00009"):
        Image.fromarray(img).save(work / "dense" / "images" / "front" / f"{stem}.png")
        Image.fromarray(img).save(work / "splat" / "eval_100" / f"{stem}.png")
    (work / "eval.json").write_text(json.dumps({"eval": ["front_00008"], "guard": []}))
    with pytest.raises(ValueError, match="1 unexpected"):
        bench.view_metrics(work, {"eval": True, "steps": 100})
    (work / "splat" / "eval_100" / "front_00009.png").unlink()
    v = bench.view_metrics(work, {"eval": True, "steps": 100})
    assert v["psnr"]["split"] == "time" and v["psnr"]["views"] == 1

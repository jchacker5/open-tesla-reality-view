import json

import numpy as np
from PIL import Image

from tsplat import bench


def _write_model(work, names):
    sp = work / "dense" / "sparse"
    sp.mkdir(parents=True)
    rows = []
    for i, n in enumerate(names, 1):
        rows += [f"{i} 1 0 0 0 0 0 0 1 {n}", ""]
    (sp / "images.txt").write_text("# header\n" + "\n".join(rows) + "\n")


def test_registration_counts_per_camera(tmp_path):
    work = tmp_path / "run"
    for cam, n in (("front", 4), ("back", 4)):
        (work / "images" / cam).mkdir(parents=True)
        for i in range(n):
            (work / "images" / cam / f"{i:05d}.jpg").touch()
    (work / "meta.json").write_text(json.dumps({"cams": ["front", "back"]}))
    _write_model(
        work, ["front/front_00000.jpg", "front/front_00001.jpg", "front/front_00002.jpg", "back/back_00000.jpg"]
    )
    r = bench.registration(work)
    assert r["front"] == {"registered": 3, "total": 4, "ratio": 0.75}
    assert r["back"]["registered"] == 1


def test_masked_psnr_ignores_masked_pixels(tmp_path):
    work = tmp_path / "run"
    gt_dir = work / "dense" / "images" / "front"
    gt_dir.mkdir(parents=True)
    (work / "dense" / "images" / "masks").mkdir()
    ev = work / "splat" / "eval_100"
    ev.mkdir(parents=True)
    gt = np.full((32, 32, 3), 128, np.uint8)
    pred = gt.copy()
    pred[:, :16] = 0  # wrong on the left half...
    mask = np.full((32, 32), 255, np.uint8)
    mask[:, :16] = 0  # ...which is masked out (ego car)
    Image.fromarray(gt).save(gt_dir / "front_00008.jpg", quality=100)
    Image.fromarray(pred).save(ev / "front_00008.png")
    Image.fromarray(mask).save(work / "dense" / "images" / "masks" / "front_00008.png")
    p = bench.psnr(work)
    assert p["step"] == 100 and p["views"] == 1
    assert p["per_cam"]["front"] > 40  # near-perfect on the kept half


def test_gate_flags_low_registration_and_bad_reprojection():
    m = {"registration": {"front": {"ratio": 0.5}, "back": {"ratio": 0.95}}, "sparse": {"reproj_px": 2.0}}
    fails = bench.gate(m)
    assert any("front" in f for f in fails) and any("reprojection" in f for f in fails)
    assert not any("back" in f for f in fails)

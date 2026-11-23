import json

import numpy as np
import pytest
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
    p = bench.psnr(work, {"eval": True, "steps": 100})
    assert p["step"] == 100 and p["views"] == 1
    assert p["per_cam"]["front"] > 40  # near-perfect on the kept half


def test_psnr_only_reads_this_runs_eval_dir(tmp_path):
    work = tmp_path / "run"
    (work / "splat" / "eval_50").mkdir(parents=True)  # a previous run's renders
    assert bench.psnr(work, {"eval": False, "steps": 50}) is None  # not an --eval run: never read
    assert bench.psnr(work, {}) is None
    with pytest.raises(FileNotFoundError, match="eval_100"):
        bench.psnr(work, {"eval": True, "steps": 100})


def test_bench_always_appends_ledger_line_for_failed_run(tmp_path, monkeypatch):
    """SfM failed: sparse/ has no model and there is no dense/. bench still logs a gated-out line."""
    monkeypatch.setattr(bench, "environment", lambda: {"chip": "test", "ram_gb": 16})
    work = tmp_path / "drive"
    (work / "images" / "front").mkdir(parents=True)
    (work / "images" / "front" / "00000.jpg").touch()
    (work / "sparse" / "0").mkdir(parents=True)
    (work / "meta.json").write_text(json.dumps({"clip": "c", "start": 0, "dur": 20, "fps": 3, "cams": ["front"]}))
    m = bench.run("drive", "broken run", workdir=str(tmp_path))
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["note"] == "broken run" and logged["config"]["cams"] == 1
    assert set(logged["errors"]) == {"registration", "sparse"}
    assert any(f.startswith("registration:") for f in logged["gate_failures"])
    assert m["gate_failures"] == logged["gate_failures"]
    assert json.loads((work / "metrics.json").read_text())["errors"] == logged["errors"]

    bench.run("missing", workdir=str(tmp_path))  # not even meta.json: still one more line
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 2


def test_gate_flags_low_registration_and_bad_reprojection():
    m = {"registration": {"front": {"ratio": 0.5}, "back": {"ratio": 0.95}}, "sparse": {"reproj_px": 2.0}}
    fails = bench.gate(m)
    assert any("front" in f for f in fails) and any("reprojection" in f for f in fails)
    assert not any("back" in f for f in fails)

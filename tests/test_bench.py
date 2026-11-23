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


def test_registration_ignores_names_missing_from_images(tmp_path):
    work = tmp_path / "run"
    (work / "images" / "front").mkdir(parents=True)
    for i in range(4):
        (work / "images" / "front" / f"{i:05d}.jpg").touch()
    (work / "meta.json").write_text(json.dumps({"cams": ["front"]}))
    _write_model(work, ["front/front_00000.jpg", "front/front_00003.jpg", "front/front_00042.jpg"])
    assert bench.registration(work)["front"] == {"registered": 2, "total": 4, "ratio": 0.5}


def test_registration_from_a_stale_model_is_an_error_not_a_pass(tmp_path, monkeypatch):
    """Model from an 18-frame prep, images/ from a 9-frame re-prep: never report 18/9 or pass the gate."""
    monkeypatch.setattr(bench, "environment", lambda: {})
    work = tmp_path / "run"
    (work / "images" / "front").mkdir(parents=True)
    for i in range(9):
        (work / "images" / "front" / f"{i:05d}.jpg").touch()
    (work / "meta.json").write_text(json.dumps({"cams": ["front"]}))
    _write_model(work, [f"front/front_{i:05d}.jpg" for i in range(18)])
    with pytest.raises(ValueError, match="stale"):
        bench.registration(work)
    m = bench.run("run", workdir=str(tmp_path))
    assert "registration" in m["errors"] and m["gate_failures"]
    assert any(f.startswith("registration:") for f in m["gate_failures"])


def test_gate_fails_ratio_above_one_and_empty_registration():
    m = {"registration": {"front": {"registered": 18, "total": 9, "ratio": 2.0}}, "sparse": {"reproj_px": 0.5}}
    assert any("> 100%" in f for f in bench.gate(m))
    assert bench.gate({"registration": {}, "sparse": {"reproj_px": 0.5}}) == ["registration: no cameras scored"]


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


def _eval_view(work, stem="front_00008", h=40, w=20, ext=".jpg"):
    """A held-out view: mid-grey ground truth (no mask file, so every pixel counts)."""
    cam = stem.rsplit("_", 1)[0]
    (work / "dense" / "images" / cam).mkdir(parents=True, exist_ok=True)
    (work / "splat" / "eval_100").mkdir(parents=True, exist_ok=True)
    gt = np.full((h, w, 3), 128, np.uint8)
    Image.fromarray(gt).save(work / "dense" / "images" / cam / f"{stem}{ext}", quality=100)
    return gt


def test_road_band_is_the_bottom_35_percent(tmp_path):
    work = tmp_path / "run"
    gt = _eval_view(work)
    pred = gt.copy()
    pred[:26] = 0  # top 65% wrong, bottom 35% (rows 26..39) right
    Image.fromarray(pred).save(work / "splat" / "eval_100" / "front_00008.png")
    v = bench.view_metrics(work, {"eval": True, "steps": 100})
    assert v["road"]["psnr"] > 40 and v["psnr"]["overall"] < 15
    assert v["lpips"] is None and v["road"]["lpips"] is None  # no LPIPS model given: skipped


def test_sky_mask_is_excluded_and_lpips_is_masked(tmp_path):
    work = tmp_path / "run"
    gt = _eval_view(work, ext=".png")  # transparent-sky runs train on .png
    pred = gt.copy()
    pred[:10] = 255  # sky rendered differently (it's transparent in training): must not count
    Image.fromarray(pred).save(work / "splat" / "eval_100" / "front_00008.png")
    sky = np.zeros((40, 20), np.uint8)
    sky[:10] = 255
    (work / "dense" / "sky").mkdir(parents=True)
    Image.fromarray(sky).save(work / "dense" / "sky" / "front_00008.png")

    def fake_lpips(p, g):
        return np.abs(p - g).mean(2)  # a per-pixel map, like lpips(spatial=True)

    v = bench.view_metrics(work, {"eval": True, "steps": 100}, fake_lpips)
    assert v["psnr"]["sky_excluded"] and v["psnr"]["overall"] > 40
    assert v["lpips"]["overall"] == 0 and v["lpips"]["per_cam"] == {"front": 0}
    assert v["road"]["lpips"] == 0


def test_floaters_card_and_seed_in_ledger(tmp_path, monkeypatch):
    from splatfix import make_ply, write_path

    from tsplat import splats

    monkeypatch.setattr(bench, "environment", lambda: {})
    work = tmp_path / "run"
    (work / "images" / "front").mkdir(parents=True)
    (work / "meta.json").write_text(json.dumps({"clip": "c", "fps": 6, "sky": "alpha", "cams": ["front"]}))
    (work / "recon_args.json").write_text(json.dumps({"steps": 100, "seed": 7, "prune": True}))
    write_path(work / "viewer" / "path.json")
    raw = make_ply([[0, -1.0, 2], [0, -1.0, 3], [8, -1.0, 3]])
    (work / "splat").mkdir()
    splats.write_ply(work / "splat" / "splat_100.ply", raw)
    splats.write_ply(work / "viewer" / "scene.ply", raw.subset(np.array([False, False, True])))
    (work / "prune.json").write_text(json.dumps({"before": 3, "tube": 2, "giant": 0, "after": 1}))
    m = bench.run("run", workdir=str(tmp_path))
    assert m["floaters"] == {"scene": 0, "splats": 1, "raw": 2, "splats_raw": 3}
    assert m["config"]["seed"] == 7 and m["config"]["sky"] == "alpha" and m["config"]["fps"] == 6
    c = bench.card(m)
    assert "floaters in ego tube: 0 of 1 splats (2 of 3 before prune)" in c
    assert "prune: 3 -> 1 splats" in c
    assert "floaters" not in m["errors"]
    logged = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[-1])
    assert logged["floaters"]["raw"] == 2 and logged["config"]["seed"] == 7

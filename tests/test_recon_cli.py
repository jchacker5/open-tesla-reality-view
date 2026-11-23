"""`otrv recon` end to end with COLMAP/Brush stubbed: every flag reaches Brush or recon_args.json (and so
the bench card's config line)."""

import argparse
import json

import pytest

from tsplat import __main__ as cli

# Options that change how recon runs, not what it produces, so they aren't recorded.
NOT_RECORDED = {"name", "live", "skip_sfm", "debug", "help"}
RECORDED_AS = {"no_rig": "rig"}


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OTRV_SKIP_DOCTOR", "1")
    work = tmp_path / "work" / "r"
    work.mkdir(parents=True)
    (work / "meta.json").write_text(json.dumps({"fps": 6, "dur": 4, "cams": ["front"]}))
    calls: dict = {}
    dense = work / "dense"
    (dense / "sparse").mkdir(parents=True)
    poses = "".join(f"{i + 1} 1 0 0 0 0 0 {i} 1 front/front_{i:05d}.jpg\n\n" for i in range(24))
    (dense / "sparse" / "images.txt").write_text("# images\n" + poses)
    monkeypatch.setattr(cli.recon, "sfm", lambda *a: work / "sparse" / "0")
    monkeypatch.setattr(cli.recon, "undistort", lambda *a: dense)
    monkeypatch.setattr(cli.recon, "snapshot_dense", lambda *a: None)
    monkeypatch.setattr(cli.recon, "export_path", lambda *a: None)
    monkeypatch.setattr(cli.colornorm, "run", lambda *a: calls.setdefault("color_norm", True))
    monkeypatch.setattr(cli.densify, "run", lambda *a, exclude: calls.setdefault("dense_init", exclude))
    monkeypatch.setattr(cli.sky, "apply", lambda d, mode: calls.setdefault("sky", mode))
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())

    def train(work_, dense_, steps, live, eval_every, extra, rules):
        calls["train"] = {"steps": steps, "eval_every": eval_every, "extra": list(extra)}

    monkeypatch.setattr(cli.recon, "train", train)
    monkeypatch.setattr(cli.collision, "run", lambda *a, **k: calls.setdefault("collision", True))
    return work, calls


def recon_options() -> list[argparse.Action]:
    ap = cli.build_parser()
    sub = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices["recon"]._actions


def clean_dests() -> set[str]:
    a = argparse.ArgumentParser()
    cli.add_clean_args(a)
    return {x.dest for x in a._actions}


def test_every_recon_flag_is_recorded_in_recon_args(stubbed):
    work, calls = stubbed
    cli.main(["recon", "r", "--color-norm", "--dense-init", "--eval"])
    rec = json.loads((work / "recon_args.json").read_text())
    clean = clean_dests()
    dests = {RECORDED_AS.get(a.dest, a.dest) for a in recon_options()} - NOT_RECORDED - clean
    missing = sorted(d for d in dests if d not in rec)
    assert not missing, f"recon flags not in recon_args.json: {missing}"
    assert rec["color_norm"] is True and rec["dense_init"] is True and "clean" in rec
    assert calls["color_norm"] and calls["dense_init"] and calls["sky"] == "alpha"
    assert calls["train"]["eval_every"] == 8 and calls["collision"]  # baked after training
    # held out at 4/3 s (frame 8 at 6 fps; 4 s is past the 24 frames), guard frames 7 and 9: none of
    # them may seed the splat, and Brush's split must land on frame 8 only
    assert calls["dense_init"] == {"front_00007", "front_00008", "front_00009"}
    split = json.loads((work / "eval.json").read_text())
    assert split["eval"] == ["front_00008"] and split["guard"] == ["front_00007", "front_00009"]


def test_recon_args_reach_brush(stubbed):
    work, calls = stubbed
    cli.main(["recon", "r", "--seed", "3", "--max-splats", "1234", "--lpips-loss-weight", "0.1"])
    extra = calls["train"]["extra"]
    pairs = dict(zip(extra[::2], extra[1::2], strict=False))
    assert pairs["--seed"] == "3" and pairs["--max-splats"] == "1234" and pairs["--lpips-loss-weight"] == "0.1"
    assert "--opac-loss-weight" in pairs and "--scale-loss-weight" in pairs
    rec = json.loads((work / "recon_args.json").read_text())
    assert rec["max_splats"] == 1234 and rec["lpips_loss_weight"] == 0.1
    assert "color_norm" not in calls  # off by default


def test_max_splats_default_follows_ram(monkeypatch):
    monkeypatch.setattr(cli.doctor, "ram_gb", lambda: 16.0)
    assert cli.default_max_splats() == 3_000_000
    monkeypatch.setattr(cli.doctor, "ram_gb", lambda: 24.0)
    assert cli.default_max_splats() == 6_000_000
    monkeypatch.setattr(cli.doctor, "ram_gb", lambda: None)
    assert cli.default_max_splats() == 3_000_000


def test_lpips_default_depends_on_the_brush():
    assert cli.lpips_weight(None, (1, 0, 0)) == 0.05  # Brush main (source build)
    assert cli.lpips_weight(None, (0, 3, 0)) == 0.0  # v0.3.0's LPIPS loss aborts on Metal
    assert cli.lpips_weight(None, None) == 0.0
    assert cli.lpips_weight(0.2, (0, 3, 0)) == 0.2  # an explicit value always wins


def test_default_lpips_reaches_brush_and_recon_args(stubbed, monkeypatch):
    work, calls = stubbed
    monkeypatch.setattr(cli.brush, "version", lambda exe: (1, 0, 0))
    cli.main(["recon", "r"])
    extra = calls["train"]["extra"]
    assert extra[extra.index("--lpips-loss-weight") + 1] == "0.05"
    assert json.loads((work / "recon_args.json").read_text())["lpips_loss_weight"] == 0.05

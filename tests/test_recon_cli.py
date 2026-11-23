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
    (work / "meta.json").write_text(json.dumps({"fps": 6, "cams": ["front"]}))
    calls: dict = {}
    dense = work / "dense"
    monkeypatch.setattr(cli.recon, "sfm", lambda *a: work / "sparse" / "0")
    monkeypatch.setattr(cli.recon, "undistort", lambda *a: dense)
    monkeypatch.setattr(cli.recon, "snapshot_dense", lambda *a: None)
    monkeypatch.setattr(cli.recon, "export_path", lambda *a: None)
    monkeypatch.setattr(cli.colornorm, "run", lambda *a: calls.setdefault("color_norm", True))
    monkeypatch.setattr(cli.densify, "run", lambda *a: calls.setdefault("dense_init", True))
    monkeypatch.setattr(cli.sky, "apply", lambda d, mode: calls.setdefault("sky", mode))
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())

    def train(work_, dense_, steps, live, eval_every, extra, rules):
        calls["train"] = {"steps": steps, "eval_every": eval_every, "extra": list(extra)}

    monkeypatch.setattr(cli.recon, "train", train)
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
    assert calls["train"]["eval_every"] == 8

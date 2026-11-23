import json

import numpy as np
import pytest
from splatfix import make_ply, path_dict, write_path

from tsplat import clean, splats

# path along +z from z=0 to z=20, road plane y = 0, up = -y; mpu = 1 unless stated


def _ego(mpu=1.0):
    return splats.EgoPath.from_dict(path_dict(mpu=mpu))


def test_prune_drops_tube_and_giant_faint_splats_only_when_dehaze_is_off():
    ego = _ego(mpu=2.0)
    pts = [[0, -0.5, 3], [10, -2, 3], [10, -2, 4], [10, -2, 5]]
    # metres: 1 m up in the tube; a 4 m faint splat; a 4 m opaque splat (kept); a 1 m faint one (kept)
    ply = make_ply(pts, opacity=[0.9, 0.1, 0.9, 0.1], scale_m=[0.1, 4.0, 4.0, 1.0], mpu=2.0)
    kept, stats = clean.clean(ply, ego, clean.Rules(dehaze=False))
    assert stats == {"before": 4, "after": 2, "removed": {"tube": 1, "giant": 1}}
    assert kept.xyz()[:, 2].tolist() == [4, 5]
    assert kept.data.dtype.names == ply.data.dtype.names


def test_each_dehaze_rule_and_first_rule_attribution():
    pts = [
        [5, -1, 5],  # 0 low opacity (0.02)
        [5, -1, 6],  # 1 large translucent: 1.5 m at 0.3
        [5, -1, 7],  # 2 needle: 0.6 x 0.01 m at 0.4
        [60, -0.3, 8],  # 3 ground fog: 60 m out, 0.3 m up
        [60, -5.0, 8],  # 4 60 m out but 5 m up: kept (a building)
        [5, -1, 9],  # 5 solid, small: kept
        [5, -1, 10],  # 6 large but opaque (0.9): kept
        [0, -1, 11],  # 7 in the ego tube AND low opacity: counted once, as tube
    ]
    ply = make_ply(pts, opacity=[0.02, 0.3, 0.4, 0.9, 0.9, 0.9, 0.9, 0.02], scale_m=0.05)
    ply.data["scale_0"][1] = np.log(1.5)
    ply.data["scale_0"][2] = np.log(0.6)
    ply.data["scale_1"][2] = ply.data["scale_2"][2] = np.log(0.01)
    ply.data["scale_0"][6] = np.log(1.5)
    kept, stats = clean.clean(ply, _ego(), clean.Rules())
    assert stats["removed"] == {
        "tube": 1,
        "giant": 0,
        "low_opacity": 1,
        "translucent": 1,
        "needle": 1,
        "ground_fog": 1,
    }
    assert sorted(kept.xyz()[:, 2].tolist()) == [8, 9, 10]
    assert stats["after"] == 3


def test_sky_fog_rule_is_opt_in_and_needs_pale_bright_above_eye_level():
    pts = [[5, -10, 10], [5, -10, 11], [5, -0.5, 12], [80, -10, 13]]
    ply = make_ply(pts, opacity=0.9)
    ply.data["f_dc_0"] = ply.data["f_dc_1"] = ply.data["f_dc_2"] = 1.0  # pale and bright (rgb 0.78)
    ply.data["f_dc_0"][1] = -1.5  # saturated (cyan-ish): not sky haze
    assert "sky_fog" not in clean.masks(ply, _ego(), clean.Rules())
    m = clean.masks(ply, _ego(), clean.Rules(sky_fog=True))["sky_fog"]
    assert m.tolist() == [True, False, False, False]  # #2 below eye level, #3 farther than 60 m


def test_translucent_mass():
    ply = make_ply([[0, 0, 0]] * 4, opacity=[0.9, 0.9, 0.1, 0.1])
    assert clean.translucent_mass(ply) == pytest.approx(0.2 / 2.0)


def test_run_writes_scene_raw_and_clean_json_and_reruns_from_raw(tmp_path):
    work = tmp_path / "w"
    write_path(work / "viewer" / "path.json")
    src = work / "splat" / "splat_100.ply"
    ply = make_ply([[0, -1, 4], [5, -1, 5], [5, -1, 6]], opacity=[0.9, 0.9, 0.02])
    splats.write_ply(src, ply)
    stats = clean.run(work, src, clean.Rules())
    assert stats["before"] == 3 and stats["after"] == 1
    assert json.loads((work / "clean.json").read_text())["removed"]["tube"] == 1
    assert len(splats.read_ply(work / "viewer" / "scene_raw.ply")) == 3
    assert len(splats.read_ply(work / "viewer" / "scene.ply")) == 1
    assert len(splats.read_ply(src)) == 3  # the trained splat is untouched (scene_raw may be a hard link)
    again = clean.run(work, work / "viewer" / "scene_raw.ply", clean.Rules(prune=False, dehaze=False))
    assert again["after"] == 3 and len(splats.read_ply(work / "viewer" / "scene.ply")) == 3
    assert stats["translucent_mass_raw"] > stats["translucent_mass"] == 0


def test_cli_clean_reruns_with_new_rules(tmp_path, monkeypatch):
    from tsplat import __main__ as cli

    monkeypatch.chdir(tmp_path)
    work = tmp_path / "work" / "d"
    (work).mkdir(parents=True)
    (work / "meta.json").write_text("{}")
    (work / "recon_args.json").write_text(json.dumps({"steps": 100, "sky": "alpha"}))
    write_path(work / "viewer" / "path.json")
    splats.write_ply(work / "viewer" / "scene_raw.ply", make_ply([[0, -1, 4], [5, -1, 5]]))
    cli.main(["clean", "d", "--no-prune", "--dehaze-sky"])
    assert len(splats.read_ply(work / "viewer" / "scene.ply")) == 2
    rec = json.loads((work / "recon_args.json").read_text())
    assert rec["clean"]["prune"] is False and rec["clean"]["sky_fog"] is False  # sky was trained transparent

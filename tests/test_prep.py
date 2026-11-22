import json

import numpy as np
import pytest

from tsplat import prep, recon
from tsplat.ingest import CAMS

FOUR = ["front", "back", "left_repeater", "right_repeater"]


def fake_images(root, cams, n):
    for c in cams:
        d = root / "images" / c
        d.mkdir(parents=True)
        for t in range(n):
            (d / f"{t:05d}.jpg").write_bytes(b"")


def read_pairs(root):
    return [tuple(line.split()) for line in (root / "pairs.txt").read_text().splitlines()]


@pytest.mark.parametrize("cams", [CAMS, FOUR], ids=["6cam", "4cam"])
def test_write_pairs(tmp_path, cams):
    fake_images(tmp_path, cams, 12)
    count = prep.write_pairs(tmp_path, cams)
    pairs = read_pairs(tmp_path)
    assert count == len(pairs) == len(set(pairs)) > 0
    assert pairs == sorted(pairs)
    nbrs = prep.neighbours(cams)
    cross = set()
    for a, b in pairs:
        assert a < b  # canonical order, so no (b, a) duplicates and no self-pairs
        ca, cb = a.split("/")[0], b.split("/")[0]
        assert ca in cams and cb in cams
        if ca != cb:
            assert cb in nbrs[ca] and ca in nbrs[cb]
            cross.add(frozenset((ca, cb)))
    # every neighbour relation actually produced pairs
    assert cross == {frozenset((c, n)) for c in cams for n in nbrs[c]}


def test_four_cam_front_pairs_with_repeaters(tmp_path):
    fake_images(tmp_path, FOUR, 12)
    prep.write_pairs(tmp_path, FOUR)
    front_partners = {
        b.split("/")[0] if a.startswith("front/") else a.split("/")[0]
        for a, b in read_pairs(tmp_path)
        if "front/" in (a[:6], b[:6])
    }
    assert {"left_repeater", "right_repeater"} <= front_partners
    assert "back" not in front_partners


def test_write_pairs_same_camera_window(tmp_path):
    fake_images(tmp_path, ["front"], 10)
    prep.write_pairs(tmp_path, ["front"], same_k=2)
    assert read_pairs(tmp_path)[:3] == [
        ("front/00000.jpg", "front/00001.jpg"),
        ("front/00000.jpg", "front/00002.jpg"),
        ("front/00001.jpg", "front/00002.jpg"),
    ]


@pytest.mark.parametrize("cams", [CAMS, FOUR], ids=["6cam", "4cam"])
def test_write_rig_config(tmp_path, cams):
    prep.write_rig_config(tmp_path, cams)
    [rig] = json.loads((tmp_path / "rig_config.json").read_text())
    entries = {e["image_prefix"].rstrip("/"): e for e in rig["cameras"]}
    assert set(entries) == set(cams)
    assert entries["front"] == {"image_prefix": "front/", "ref_sensor": True}
    assert sum(bool(e.get("ref_sensor")) for e in entries.values()) == 1

    back = entries["back"]
    R = recon.qvec_to_R(back["cam_from_rig_rotation"])
    assert np.allclose(R, np.diag([-1.0, 1.0, -1.0]), atol=1e-9)  # 180 degree yaw
    p = np.array(prep.RIG["back"][1])
    assert np.allclose(back["cam_from_rig_translation"], -R @ p, atol=1e-4)

    for cam, e in entries.items():
        if cam == "front":
            continue
        R = recon.qvec_to_R(e["cam_from_rig_rotation"])
        yaw = np.radians(prep.RIG[cam][0])
        # the camera's optical axis, expressed in the rig (front camera) frame, maps to +z
        assert np.allclose(R @ [np.sin(yaw), 0, np.cos(yaw)], [0, 0, 1], atol=1e-9)
        assert np.allclose(e["cam_from_rig_translation"], -R @ np.array(prep.RIG[cam][1]), atol=1e-4)

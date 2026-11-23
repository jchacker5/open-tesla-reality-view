import json

import numpy as np
import pytest
from PIL import Image

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


def test_write_rig_config_without_front_uses_back_frame(tmp_path):
    """No front camera: back is the reference, so offsets must be expressed in the back camera's frame.
    Back sits at (0, 0.55, -3) and looks backwards (x_back = -x_front, z_back = -z_front). The left
    repeater sits at (-0.95, 0.55, 0.9): 0.95 m to the car's left = the back camera's right (+x), and
    3.9 m towards the car's front = behind the back camera (-z). Same height, so y = 0."""
    cams = ["back", "left_repeater", "right_repeater"]
    prep.write_rig_config(tmp_path, cams)
    [rig] = json.loads((tmp_path / "rig_config.json").read_text())
    entries = {e["image_prefix"].rstrip("/"): e for e in rig["cameras"]}
    assert entries["back"] == {"image_prefix": "back/", "ref_sensor": True}
    expected_centre = {"left_repeater": [0.95, 0.0, -3.9], "right_repeater": [-0.95, 0.0, -3.9]}
    for cam, centre in expected_centre.items():
        e = entries[cam]
        R = recon.qvec_to_R(e["cam_from_rig_rotation"])
        t = np.array(e["cam_from_rig_translation"])
        assert np.allclose(-R.T @ t, centre, atol=1e-4)  # camera centre in the rig (back) frame
        # relative yaw: repeater yaw - back yaw (left: -140 - 180 = -320 = +40 degrees)
        yaw = np.radians(prep.RIG[cam][0] - prep.RIG["back"][0])
        assert np.allclose(R @ [np.sin(yaw), 0, np.cos(yaw)], [0, 0, 1], atol=1e-9)


def test_telemetry_samples_carry_exact_time():
    rows = [{"frame_seq_no": 100 + i, "vehicle_speed_mps": 10.0} for i in range(36 * 3)]
    tel = prep.telemetry(rows, 36.0, start=1.0, dur=1.0)
    ts = [s["t"] for s in tel]
    assert ts[0] == 0 and ts[-1] <= 1.0 and all(b - a >= 0.1 - 1e-6 for a, b in zip(ts, ts[1:], strict=False))
    assert prep.telemetry([], 36.0, 0, 1) == []


CLIP = {"id": "2026-01-02_03-04-05", "timestamp": "2026-01-02_03-04-05", "seconds": 60.0}


@pytest.mark.parametrize(
    ("start", "dur", "msg"),
    [(-1, 20, "must be >= 0"), (10, 0, "must be >= 0"), (60, 5, "past the end"), (50, 20, "Use --dur 10")],
)
def test_check_window_rejects_bad_windows(start, dur, msg):
    with pytest.raises(SystemExit, match=msg):
        prep.check_window(CLIP, start, dur)


def test_check_window_accepts_fitting_windows():
    prep.check_window(CLIP, 40, 20)
    prep.check_window(CLIP, 40, 20.3)  # within slack for clips a few frames short of 60 s
    prep.check_window(CLIP | {"seconds": None}, 100, 20)  # unknown length: let ffmpeg decide


def test_run_clears_stale_outputs_and_records_actual_dur(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    clip = CLIP | {"kind": "RecentClips", "cams": {c: f"/x/{c}.mp4" for c in ("front", "back")}}
    (data / "index.json").write_text(json.dumps([clip]))
    work = tmp_path / "work"
    # leftovers from an earlier, longer, 6-camera prep
    for stale in ("images/front/00099.jpg", "images/left_pillar/00000.jpg", "masks/left_pillar/00000.jpg.png"):
        (work / "drive" / stale).parent.mkdir(parents=True, exist_ok=True)
        (work / "drive" / stale).write_bytes(b"old")
    (work / "drive" / "viewer" / "cams").mkdir(parents=True)
    (work / "drive" / "viewer" / "cams" / "left_pillar.mp4").write_bytes(b"old")

    def fake_extract(cams, out, start, dur, fps, width):  # front gets 12 frames, back only 11
        for cam in cams:
            d = out / "images" / cam
            d.mkdir(parents=True, exist_ok=True)
            for i in range(12 if cam == "front" else 11):
                Image.fromarray(np.full((8, 8, 3), i * 9, np.uint8)).save(d / f"{i:05d}.jpg")

    seen = {}
    monkeypatch.setattr(prep, "extract_frames", fake_extract)
    monkeypatch.setattr(prep, "write_viewer_assets", lambda cams, out, start, dur: seen.update(dur=dur))
    out = prep.run(str(data / "index.json"), clip["id"], "drive", 10, 4, 3.0, 640, False, workdir=str(work))

    assert sorted(p.name for p in (out / "images").iterdir()) == ["back", "front"]
    assert sorted(p.name for p in (out / "masks").iterdir()) == ["back", "front"]
    assert not (out / "images" / "front" / "00099.jpg").exists()
    assert not (out / "viewer" / "cams").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["dur"] == pytest.approx(11 / 3.0, abs=1e-3) and meta["requested_dur"] == 4
    assert meta["frames"] == {"front": 12, "back": 11}
    assert seen["dur"] == meta["dur"]


def fake_recon(work, n):
    """What a finished `otrv recon --eval` leaves behind for n frames per camera."""
    sp = work / "dense" / "sparse"
    sp.mkdir(parents=True)
    rows = []
    for i in range(n):
        rows += [f"{i + 1} 1 0 0 0 0 0 0 1 front/front_{i:05d}.jpg", ""]
        rows += [f"{n + i + 1} 1 0 0 0 0 0 0 1 back/back_{i:05d}.jpg", ""]
    (sp / "images.txt").write_text("\n".join(rows) + "\n")
    (work / "sparse" / "0").mkdir(parents=True)
    (work / "sparse" / "0" / "images.bin").write_bytes((2 * n).to_bytes(8, "little"))
    (work / "splat" / "eval_100").mkdir(parents=True)
    (work / "nvs").mkdir()
    for f in ("colmap.db", "timings.json", "metrics.json", "viewer/scene.ply", "viewer/path.json"):
        (work / f).write_text("{}")
    (work / "recon_args.json").write_text(json.dumps({"steps": 100, "eval": True}))


def test_reprep_deletes_downstream_state(tmp_path, monkeypatch):
    """Repro: prep 6 s -> recon -> re-prep 3 s under the same name. Before the fix, bench scored the old
    model against the new frames ("front 18/9") and passed; view showed the old splat."""
    from tsplat import __main__ as cli
    from tsplat import bench

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OTRV_SKIP_DOCTOR", "1")
    monkeypatch.setattr(bench, "environment", lambda: {})
    data = tmp_path / "data"
    data.mkdir()
    clip = CLIP | {"kind": "RecentClips", "cams": {c: f"/x/{c}.mp4" for c in ("front", "back")}}
    (data / "index.json").write_text(json.dumps([clip]))
    frames = {"n": 18}

    def fake_extract(cams, out, start, dur, fps, width):
        for cam in cams:
            d = out / "images" / cam
            d.mkdir(parents=True, exist_ok=True)
            for i in range(frames["n"]):
                Image.fromarray(np.full((8, 8, 3), i * 9, np.uint8)).save(d / f"{i:05d}.jpg")

    def viewer_assets(cams, out, start, dur):
        (out / "viewer").mkdir(parents=True, exist_ok=True)
        (out / "viewer" / "telemetry.json").write_text("[]")

    monkeypatch.setattr(prep, "extract_frames", fake_extract)
    monkeypatch.setattr(prep, "write_viewer_assets", viewer_assets)
    index = str(data / "index.json")
    out = prep.run(index, clip["id"], "drive", 0, 6, 3.0, 640, False)
    fake_recon(out, 18)

    frames["n"] = 9
    prep.run(index, clip["id"], "drive", 0, 3, 3.0, 640, False)
    downstream = ("sparse", "dense", "splat", "nvs", "colmap.db", "recon_args.json", "timings.json", "metrics.json")
    for gone in downstream:
        assert not (out / gone).exists(), gone
    assert not (out / "viewer" / "scene.ply").exists() and not (out / "viewer" / "path.json").exists()
    assert (out / "meta.json").exists() and len(list((out / "images" / "front").iterdir())) == 9

    m = bench.run("drive", workdir="work")  # scored directly: nothing to score, so it must FAIL
    assert m["gate_failures"] and "registration" in m["errors"]

    def refused(*argv):
        with pytest.raises(SystemExit) as e:
            cli.main(list(argv))
        return str(e.value.code)

    assert "run `otrv recon drive` first" in refused("view", "drive")
    assert "run `otrv recon drive` first" in refused("bench", "drive")
    assert "run `otrv recon drive` first" in refused("recon", "drive", "--skip-sfm")

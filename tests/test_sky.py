import json

import numpy as np
import pytest
from PIL import Image

from tsplat import recon, sky


def street(h=120, w=160, seed=0, building=True):
    """Smooth bright sky gradient over the top 40%, textured dark ground below, and a dark textured
    building rising into the sky in the middle (it touches the top edge but must not be sky)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    rows = int(h * 0.4)
    img[:rows] = np.linspace(230, 190, rows)[:, None, None] * np.array([0.85, 0.95, 1.0])
    img[rows:] = 90 + rng.normal(0, 30, (h - rows, w, 3))
    if building:
        img[:rows, 60:90] = 60 + rng.normal(0, 30, (rows, 30, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_candidate_finds_top_connected_sky_but_not_the_building():
    m = sky.candidate(street().astype(np.float32))
    h = m.shape[0]
    assert m[: int(h * 0.3), :50].mean() > 0.9 and m[: int(h * 0.3), 100:].mean() > 0.9
    assert m[:, 62:88].mean() < 0.05  # the building
    assert m[int(h * 0.5) :].sum() == 0  # never below the upper half


def test_candidate_ignores_bright_smooth_regions_not_touching_the_top():
    img = np.full((120, 160, 3), 40, np.uint8)
    img[30:55, 20:140] = 220  # a bright smooth band in the upper half, but enclosed by dark
    assert sky.candidate(img.astype(np.float32)).sum() == 0


def test_heuristic_temporal_median_drops_a_one_frame_glitch(tmp_path):
    frames = []
    for i in range(7):
        im = street(seed=i)
        if i == 3:
            im[:48, 60:90] = 225  # glare: the building looks like sky for one frame
        p = tmp_path / f"{i:05d}.jpg"
        Image.fromarray(im).save(p, quality=95)
        frames.append(p)
    assert sky.candidate(np.asarray(Image.open(frames[3]), np.float32))[:20, 65:85].mean() > 0.5
    keep = np.ones((120, 160), bool)
    keep[:, 150:] = False  # ego body on the right edge: never sky
    masks = sky.heuristic(frames, keep)
    assert len(masks) == 7 and masks[0].shape == (120, 160)
    assert masks[3][:, 64:86].mean() < 0.05  # outvoted by its neighbours
    assert masks[3][:30, 5:45].mean() > 0.9
    assert not masks[0][:, 150:].any()


def test_colours_top_and_horizon(tmp_path):
    im = np.zeros((100, 10, 3), np.uint8)
    im[:25] = (10, 20, 250)  # top of the sky: blue
    im[25:50] = (200, 200, 200)  # horizon: light grey
    p = tmp_path / "a.jpg"
    Image.fromarray(im).save(p, quality=100)
    m = np.zeros((100, 10), bool)
    m[:50] = True
    c = sky.colours([p], [m], stride=1)
    assert c["top"][2] > 0.9 and c["top"][0] < 0.1
    assert c["horizon"] == pytest.approx([0.784] * 3, abs=0.02)
    assert sky.colours([p], [np.zeros((100, 10), bool)]) is None


def test_write_produces_masks_and_sky_json(tmp_path):
    out = tmp_path / "w"
    for cam in ("front", "back"):
        (out / "images" / cam).mkdir(parents=True)
        for i in range(4):
            Image.fromarray(street(seed=i)).save(out / "images" / cam / f"{i:05d}.jpg", quality=95)
    info = sky.write(out, ["front", "back"])
    assert (out / "sky" / "front" / "00002.jpg.png").exists()
    assert json.loads((out / "sky.json").read_text()) == info
    assert info["segmenter"] == "heuristic" and info["share"]["front"] > 0.2
    assert info["colours"]["top"][2] > info["colours"]["top"][0]  # bluish


def _dense(tmp_path, partial_mask: bool):
    """Two undistorted images; front_00001 has an ego mask with ignored pixels if partial_mask."""
    dense = tmp_path / "dense"
    (dense / "images" / "front").mkdir(parents=True)
    (dense / "images" / "masks").mkdir()
    (dense / "sky").mkdir()
    (dense / "sparse").mkdir()
    lines = ["# images"]
    for i in range(2):
        stem = f"front_{i:05d}"
        Image.fromarray(street(seed=i)).save(dense / "images" / "front" / f"{stem}.jpg", quality=95)
        s = np.zeros((120, 160), np.uint8)
        s[:30] = 255
        Image.fromarray(s).save(dense / "sky" / f"{stem}.png")
        keep = np.full((120, 160), 255, np.uint8)
        if partial_mask and i == 1:
            keep[100:] = 0
        Image.fromarray(keep).save(dense / "images" / "masks" / f"{stem}.png")
        lines += [f"{i + 1} 1 0 0 0 0 0 0 1 front/{stem}.jpg", ""]
    (dense / "sparse" / "images.txt").write_text("\n".join(lines) + "\n")
    return dense


def test_apply_alpha_makes_unmasked_images_transparent_and_masked_ones_ignore_sky(tmp_path):
    dense = _dense(tmp_path, partial_mask=True)
    counts = sky.apply(dense, "alpha")
    assert counts == {"transparent": 1, "ignored": 1, "untouched": 0}
    rgba = Image.open(dense / "images" / "front" / "front_00000.png")
    assert rgba.mode == "RGBA"
    a = np.asarray(rgba)[..., 3]
    assert (a[:30] == 0).all() and (a[30:] == 255).all()
    assert not (dense / "images" / "front" / "front_00000.jpg").exists()
    assert not (dense / "images" / "masks" / "front_00000.png").exists()  # else Brush would use it, not alpha
    m = np.asarray(Image.open(dense / "images" / "masks" / "front_00001.png"))
    assert (m[:30] == 0).all() and (m[100:] == 0).all() and (m[30:100] == 255).all()
    names = [line.split()[-1] for line in (dense / "sparse" / "images.txt").read_text().splitlines()[1::2]]
    assert names == ["front/front_00000.png", "front/front_00001.jpg"]


@pytest.mark.parametrize("mode", ["ignore", "off"])
def test_apply_ignore_and_off(tmp_path, mode):
    dense = _dense(tmp_path, partial_mask=False)
    counts = sky.apply(dense, mode)
    assert counts["transparent"] == 0
    m = np.asarray(Image.open(dense / "images" / "masks" / "front_00000.png"))
    assert (m[:30] == 0).all() if mode == "ignore" else (m == 255).all()
    assert (dense / "images" / "front" / "front_00000.jpg").exists()


def test_restore_dense_undoes_levers(tmp_path):
    work = tmp_path
    dense = _dense(tmp_path, partial_mask=False)
    recon.snapshot_dense(work)
    sky.apply(dense, "alpha")
    assert not (dense / "images" / "front" / "front_00000.jpg").exists()
    recon.restore_dense(work)
    assert (dense / "images" / "front" / "front_00000.jpg").exists()
    assert "front_00000.jpg" in (dense / "sparse" / "images.txt").read_text()


def test_export_path_carries_sky_colours(tmp_path):
    from test_recon import make_work

    work = make_work(tmp_path)
    (work / "sky.json").write_text(json.dumps({"colours": {"top": [0.1, 0.2, 0.9], "horizon": [0.8, 0.8, 0.8]}}))
    path = json.loads(recon.export_path(work, work / "dense").read_text())
    assert path["sky"] == {"top": [0.1, 0.2, 0.9], "horizon": [0.8, 0.8, 0.8]}

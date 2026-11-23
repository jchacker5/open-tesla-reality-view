import numpy as np
from PIL import Image

from tsplat import colornorm


def test_lab_round_trip():
    rgb = np.random.default_rng(0).uniform(0, 1, (50, 3))
    assert np.allclose(colornorm.lab_to_rgb(colornorm.rgb_to_lab(rgb)), rgb, atol=1e-6)
    white = colornorm.rgb_to_lab(np.ones(3))
    assert np.allclose(white, [100, 0, 0], atol=1e-3)


def _scene(rng, h=40, w=60):
    """Same-ish street content for every camera: a random texture around mid grey."""
    return np.clip(0.45 + rng.normal(0, 0.12, (h, w, 3)), 0, 1)


def test_cameras_are_matched_to_the_front_camera(tmp_path):
    dense = tmp_path / "dense"
    rng = np.random.default_rng(1)
    for cam, tint in (("front", (1.0, 1.0, 1.0)), ("back", (0.75, 0.8, 0.95))):  # back: darker and bluer
        (dense / "images" / cam).mkdir(parents=True)
        for i in range(8):
            img = _scene(rng) * np.array(tint)
            Image.fromarray((img * 255).astype(np.uint8)).save(
                dense / "images" / cam / f"{cam}_{i:05d}.jpg", quality=98
            )
    before = np.asarray(Image.open(dense / "images" / "front" / "front_00003.jpg"))
    report = colornorm.run(dense, fps=2.0)
    assert np.array_equal(np.asarray(Image.open(dense / "images" / "front" / "front_00003.jpg")), before)
    assert report["back"]["shift"][0] > 3  # brightened

    def mean_lab(cam):
        ims = [np.asarray(Image.open(p), np.float64) / 255 for p in sorted((dense / "images" / cam).glob("*.jpg"))]
        return colornorm.rgb_to_lab(np.stack(ims)).reshape(-1, 3).mean(0)

    assert np.allclose(mean_lab("back"), mean_lab("front"), atol=1.5)


def test_masked_and_sky_pixels_do_not_drive_the_statistics(tmp_path):
    dense = tmp_path / "dense"
    rng = np.random.default_rng(2)
    (dense / "images" / "masks").mkdir(parents=True)
    (dense / "sky").mkdir()
    for cam in ("front", "left_repeater"):
        (dense / "images" / cam).mkdir()
        for i in range(4):
            img = _scene(rng)
            if cam == "left_repeater":
                img[:, :20] = 1.0  # the car's white body, masked out
            Image.fromarray((img * 255).astype(np.uint8)).save(
                dense / "images" / cam / f"{cam}_{i:05d}.jpg", quality=98
            )
            keep = np.full((40, 60), 255, np.uint8)
            if cam == "left_repeater":
                keep[:, :20] = 0
            Image.fromarray(keep).save(dense / "images" / "masks" / f"{cam}_{i:05d}.png")
    report = colornorm.run(dense, fps=2.0)
    assert abs(report["left_repeater"]["shift"][0]) < 2  # the white body would have darkened everything

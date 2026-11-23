import hashlib

import numpy as np
import pytest
from PIL import Image

from tsplat import enhance

# ---- license gate ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "env", "ok"),
    [
        (False, {}, False),
        (True, {}, True),
        (False, {"OTRV_ACCEPT_NVIDIA_LICENSE": "1"}, True),
        (False, {"OTRV_ACCEPT_NVIDIA_LICENSE": "yes"}, True),
        (False, {"OTRV_ACCEPT_NVIDIA_LICENSE": "0"}, False),
        (False, {"OTRV_ACCEPT_NVIDIA_LICENSE": ""}, False),
    ],
)
def test_license_accepted(flag, env, ok):
    assert enhance.license_accepted(flag, env) is ok


def test_require_license_refuses_with_notice():
    with pytest.raises(SystemExit) as e:
        enhance.require_license(False, {})
    msg = str(e.value.code)
    assert "NVIDIA License" in msg and "non-commercial" in msg and "Stability AI Community License" in msg
    assert "--accept-nvidia-license" in msg and "OTRV_ACCEPT_NVIDIA_LICENSE=1" in msg
    assert "nothing was downloaded" in msg


def test_require_license_accepts(capsys):
    enhance.require_license(False, {"OTRV_ACCEPT_NVIDIA_LICENSE": "1"})
    assert "OTRV_ACCEPT_NVIDIA_LICENSE" in capsys.readouterr().out
    enhance.require_license(True, {})
    assert "--accept-nvidia-license" in capsys.readouterr().out


# ---- pinned pipeline download ------------------------------------------------------------------------


def test_fetch_pipeline_verifies_and_caches(tmp_path):
    src = tmp_path / "pipeline_difix.py"
    src.write_text("class DifixPipeline:\n    pass\n")
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    got = enhance.fetch_pipeline(cache, src.as_uri(), digest)
    assert got.read_bytes() == src.read_bytes() and got.parent == cache
    src.unlink()  # cached: no second download
    assert enhance.fetch_pipeline(cache, src.as_uri(), digest) == got
    assert enhance._import_pipeline(got).__name__ == "DifixPipeline"


def test_fetch_pipeline_rejects_bad_checksum(tmp_path):
    src = tmp_path / "pipeline_difix.py"
    src.write_text("import os  # tampered\n")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        enhance.fetch_pipeline(tmp_path / "cache", src.as_uri(), "0" * 64)
    assert not list((tmp_path / "cache").iterdir())  # nothing kept, nothing imported


def test_pinned_pipeline_is_a_commit_and_sha256():
    assert len(enhance.PIPELINE_COMMIT) == 40 and enhance.PIPELINE_COMMIT in enhance.PIPELINE_URL
    assert len(enhance.PIPELINE_SHA256) == 64


# ---- reference frame -------------------------------------------------------------------------------


@pytest.fixture
def front(tmp_path):
    d = tmp_path / "images" / "front"
    d.mkdir(parents=True)
    for i in range(8):  # 2 s at 4 fps
        (d / f"{i:05d}.jpg").write_bytes(b"")
    (d / "._00003.jpg").write_bytes(b"")  # macOS AppleDouble junk is ignored
    return d


@pytest.mark.parametrize(
    ("t", "want"),
    [(0.0, 0), (0.1, 0), (0.13, 1), (0.25, 1), (1.0, 4), (1.74, 7), (99.0, 7), (-3.0, 0)],
)
def test_nearest_reference(front, t, want):
    assert enhance.nearest_reference(front, 4.0, t).name == f"{want:05d}.jpg"


def test_nearest_reference_tie_prefers_earlier(front):
    assert enhance.nearest_reference(front, 4.0, 0.125).name == "00000.jpg"


def test_nearest_reference_empty(tmp_path):
    assert enhance.nearest_reference(tmp_path, 4.0, 1.0) is None


@pytest.mark.parametrize(
    ("size", "want"), [((1920, 1080), (1024, 576)), ((1080, 1920), (1024, 1824)), ((1080, 1080), (1024, 1024))]
)
def test_work_size_multiple_of_8(size, want):
    w, h = enhance.work_size(*size)
    assert (w, h) == want and w % 8 == 0 and h % 8 == 0


# ---- enhance loop (fake model) -----------------------------------------------------------------------


def test_enhance_frames_resumes_and_passes_refs(tmp_path):
    src = []
    for i in range(4):
        p = tmp_path / "splat" / f"{i:05d}.png"
        p.parent.mkdir(exist_ok=True)
        Image.new("RGB", (16, 8), (i * 10, 0, 0)).save(p)
        src.append(p)
    out = tmp_path / "difix"
    out.mkdir()
    Image.new("RGB", (16, 8), (1, 2, 3)).save(out / "00001.png")  # already done
    calls = []

    def model(img, ref):
        calls.append((img.getpixel((0, 0))[0], ref))
        return Image.new("RGB", img.size, (255, 255, 255))

    outs = enhance.enhance_frames(src, out, model, ref_for=lambda i: None)
    assert [p.name for p in outs] == ["00000.png", "00001.png", "00002.png", "00003.png"]
    assert [c[0] for c in calls] == [0, 20, 30]  # frame 1 skipped
    assert Image.open(out / "00001.png").getpixel((0, 0)) == (1, 2, 3)
    assert not list(out.glob("*.part.png"))


# ---- stabilisation math ----------------------------------------------------------------------------


def _shift_flow(h, w, dx, dy):
    f = np.zeros((h, w, 2), np.float32)
    f[..., 0], f[..., 1] = dx, dy
    return f


def test_warp_integer_shift():
    img = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
    out = enhance.warp(img, _shift_flow(5, 6, 1, 0))  # sample one pixel to the right
    assert np.allclose(out[:, :-1], img[:, 1:])
    rgb = np.stack([img, img * 2, img * 3], -1)
    assert np.allclose(enhance.warp(rgb, _shift_flow(5, 6, 0, 1))[:-1], rgb[1:])


def test_fb_error_consistent_translation_is_zero():
    fwd, bwd = _shift_flow(8, 8, 2, -1), _shift_flow(8, 8, -2, 1)
    assert np.allclose(enhance.fb_error(fwd, bwd), 0)
    assert np.allclose(enhance.fb_error(fwd, _shift_flow(8, 8, 2, 1)), 4)  # (2, 1) + (2, -1): 4 px off


def test_stabilize_pair_blends_only_where_consistent():
    h, w = 6, 8
    prev = np.full((h, w, 3), 100, np.uint8)
    cur = np.full((h, w, 3), 200, np.uint8)
    zero = _shift_flow(h, w, 0, 0)
    out = enhance.stabilize_pair(prev, cur, zero, zero, strength=0.3)
    assert out.dtype == np.uint8 and np.all(out == 170)  # 0.3*100 + 0.7*200
    bwd = zero.copy()
    bwd[:, w // 2 :] = 3.0  # right half: backward flow with no matching forward flow -> inconsistent
    out = enhance.stabilize_pair(prev, cur, zero, bwd, strength=0.3)
    assert np.all(out[:, : w // 2] == 170) and np.all(out[:, w // 2 :] == 200)
    assert np.array_equal(enhance.stabilize_pair(prev, cur, zero, zero, strength=0.0), cur)


def test_stabilize_pair_follows_motion():
    # previous frame has a bright column at x=2; the camera moved so it sits at x=3 now
    prev = np.zeros((4, 8, 3), np.uint8)
    prev[:, 2] = 250
    cur = np.zeros((4, 8, 3), np.uint8)
    cur[:, 3] = 150
    fwd, bwd = _shift_flow(4, 8, 1, 0), _shift_flow(4, 8, -1, 0)
    out = enhance.stabilize_pair(prev, cur, fwd, bwd, strength=0.3)
    assert out[0, 3, 0] == round(0.3 * 250 + 0.7 * 150)  # aligned, not smeared to x=2
    assert out[0, 2, 0] == 0


def test_stabilize_sequence_uses_raw_flow():
    raw = [np.zeros((4, 4, 3), np.uint8)] * 3
    enh = [np.full((4, 4, 3), v, np.uint8) for v in (0, 100, 100)]
    seen = []

    def flow(a, b):
        seen.append((a is raw[0] or a is raw[1], b.shape))
        return np.zeros((4, 4, 2), np.float32)

    out = list(enhance.stabilize(raw, enh, 0.3, flow=flow))
    assert np.array_equal(out[0], enh[0])
    assert np.all(out[1] == 70)  # 0.3 * 0 + 0.7 * 100 (previous *enhanced* frame, not previous output)
    assert np.all(out[2] == 100)
    assert len(seen) == 4  # forward + backward per frame after the first


def test_farneback_translation():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    base = (rng.random((64, 96)) * 255).astype(np.uint8)
    from scipy.ndimage import gaussian_filter

    base = gaussian_filter(base.astype(np.float32), 2).astype(np.uint8)
    a = np.stack([base] * 3, -1)
    b = np.roll(a, 2, axis=1)  # content moves 2 px right
    f = enhance.farneback(a, b)
    assert abs(np.median(f[8:-8, 8:-8, 0]) - 2) < 0.5 and abs(np.median(f[8:-8, 8:-8, 1])) < 0.5

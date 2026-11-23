import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tsplat import __main__ as cli
from tsplat import render

# ---- options ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "aspect", "want"),
    [
        (None, None, ((1920, 1080), "16:9")),
        (None, "9:16", ((1080, 1920), "9:16")),
        (None, "1:1", ((1080, 1080), "1:1")),
        ("1280x720", None, ((1280, 720), "16:9")),
        ("1280x720", "16:9", ((1280, 720), "16:9")),
        ("720x1280", None, ((720, 1280), "9:16")),
    ],
)
def test_resolve_size(size, aspect, want):
    assert render.resolve_size(size, aspect) == want


@pytest.mark.parametrize(
    ("size", "aspect", "err"),
    [("1920x1080", "1:1", "is 16:9"), ("1921x1080", None, "even"), ("big", None, "1920x1080"), (None, "4:3", "4:3")],
)
def test_resolve_size_errors(size, aspect, err):
    with pytest.raises(SystemExit, match=err):
        render.resolve_size(size, aspect)


def test_frame_times():
    assert render.frame_times(0, 1, 4) == [0, 0.25, 0.5, 0.75]
    assert render.frame_times(2, 3, 30)[-1] == pytest.approx(2 + 29 / 30)
    assert len(render.frame_times(0, 20, 30)) == 600
    with pytest.raises(SystemExit, match="not after"):
        render.frame_times(5, 5, 30)


def test_viewer_url_carries_the_contract():
    url = render.viewer_url(1234, "9:16")
    assert url.startswith("http://127.0.0.1:1234/index.html?") and "render=1" in url
    assert "aspect=9%3A16" in url and "cam" not in url  # the camera goes to frame(), not the URL
    assert render.CAMS == ("chase", "driver", "free", "drone")


def test_chromium_args():
    assert "--use-angle=metal" in render.chromium_args(False, "darwin")
    assert "--use-angle=metal" not in render.chromium_args(False, "linux")
    sw = render.chromium_args(True, "darwin")
    assert "--use-angle=swiftshader" in sw and "--enable-unsafe-swiftshader" in sw and "--use-angle=metal" not in sw


# ---- pixels ----------------------------------------------------------------------------------------


def test_composite_alpha():
    base = np.full((2, 3, 3), 100, np.uint8)
    ov = np.zeros((2, 3, 4), np.uint8)
    ov[..., :3] = 200
    ov[0, 0, 3], ov[0, 1, 3], ov[0, 2, 3] = 0, 255, 128
    out = render.composite(base, ov)
    assert out.dtype == np.uint8 and out.shape == (2, 3, 3)
    assert tuple(out[0, 0]) == (100,) * 3  # transparent: base
    assert tuple(out[0, 1]) == (200,) * 3  # opaque: overlay
    assert tuple(out[0, 2]) == (round(200 * 128 / 255 + 100 * 127 / 255),) * 3
    assert np.all(out[1] == 100)


def test_label_only_touches_bottom_right():
    img = np.full((360, 640, 3), 40, np.uint8)
    out = render.draw_label(img)
    changed = np.argwhere(np.any(out != img, axis=-1))
    assert len(changed)
    assert changed[:, 0].min() > 360 * 0.8 and changed[:, 1].min() > 640 * 0.6


def test_load_png_flattens_alpha_and_resizes(tmp_path):
    import io

    buf = io.BytesIO()
    Image.new("RGBA", (20, 10), (255, 0, 0, 0)).save(buf, "PNG")
    im = render.load_png(buf.getvalue(), (40, 20), "RGB")
    assert im.size == (40, 20) and im.getpixel((5, 5)) == (0, 0, 0)
    assert render.load_png(buf.getvalue(), (20, 10), "RGBA").getpixel((0, 0)) == (255, 0, 0, 0)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_encode_h264(tmp_path):
    frames = [np.full((48, 64, 3), v, np.uint8) for v in (0, 128, 255)]
    out = tmp_path / "x.mp4"
    assert render.encode(iter(frames), out, 3, (64, 48)) == 3
    probe = _probe(out)
    assert probe["codec_name"] == "h264" and probe["pix_fmt"] == "yuv420p"
    assert (probe["width"], probe["height"]) == (64, 48) and int(probe["nb_read_frames"]) == 3


def _probe(path):
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe not installed")
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt,width,height,nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return json.loads(r.stdout)["streams"][0]


# ---- CLI guards ------------------------------------------------------------------------------------


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OTRV_ACCEPT_NVIDIA_LICENSE", raising=False)
    w = tmp_path / "work" / "a"
    (w / "viewer").mkdir(parents=True)
    (w / "meta.json").write_text(json.dumps({"fps": 4.0}))
    (w / "viewer" / "scene.ply").write_bytes(b"ply\n")
    (w / "viewer" / "path.json").write_text("{}")
    return w


def _run(*argv):
    with pytest.raises(SystemExit) as e:
        cli.main(list(argv))
    return str(e.value.code)


def test_render_needs_recon(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "run `otrv prep <clip> --name a` first" in _run("render", "a")
    (tmp_path / "work" / "a").mkdir(parents=True)
    (tmp_path / "work" / "a" / "meta.json").write_text("{}")
    assert "run `otrv recon a` first" in _run("render", "a")


def test_render_enhance_stops_at_license_gate_before_capture(work, monkeypatch):
    monkeypatch.setattr(render, "capture", lambda *a: pytest.fail("captured before the license was accepted"))
    msg = _run("render", "a", "--enhance", "difix")
    assert "NVIDIA License" in msg and "--accept-nvidia-license" in msg
    assert not (work / "render").exists()


def test_render_difix_ref_needs_camera_frames_before_capture(work, monkeypatch):
    monkeypatch.setattr(render, "capture", lambda *a: pytest.fail("captured without reference frames"))
    msg = _run("render", "a", "--enhance", "difix-ref", "--accept-nvidia-license")
    assert "needs the real camera frames" in msg and "images/front" in msg


def test_render_needs_viewer_v2_before_capture(work, monkeypatch):
    """A viewer page without the render API (viewer v1) fails at once, before a browser is launched."""
    monkeypatch.setattr(render, "capture", lambda *a: pytest.fail("captured with a viewer that has no render API"))
    with pytest.raises(SystemExit, match="needs viewer v2"):
        render.run(work, b"<html>viewer v1</html>", render.Options(out=work / "x.mp4"))
    msg = _run("render", "a", "--viewer", str(_page(work, b"<html>old viewer</html>")))
    assert "needs viewer v2" in msg and "window.otrvRender" in msg


def _page(work, html: bytes):
    p = work / "page.html"
    p.write_bytes(html)
    return p


def test_render_flags_reach_options(work, monkeypatch):
    got = {}
    monkeypatch.setattr(render, "run", lambda w, page, opts: got.update(work=w, opts=opts))
    cli.main(["render", "a", "--aspect", "9:16", "--cam", "driver", "--enhance", "difix-ref", "--end", "4"])
    o = got["opts"]
    assert o.size == (1080, 1920) and o.aspect == "9:16" and o.cam == "driver" and o.end == 4
    assert o.stabilize is True and o.label is True and o.out == Path("work/a/render.mp4")
    cli.main(["render", "a", "--enhance", "difix", "--no-stabilize", "--no-label", "--out", "x.mp4"])
    assert got["opts"].stabilize is False and got["opts"].label is False and got["opts"].out == Path("x.mp4")
    cli.main(["render", "a"])
    assert got["opts"].stabilize is False and got["opts"].enhance == "none"
    assert "between 0 and 1" in _run("render", "a", "--stabilize-strength", "2")


# ---- the real capture loop against a fake viewer page -----------------------------------------------

FAKE_VIEWER = b"""<!doctype html><html><body style="margin:0;background:#fff">
<canvas id="c"></canvas>
<script>
const c = document.getElementById('c');
c.width = innerWidth; c.height = innerHeight;
const g = c.getContext('2d');
const p = new URLSearchParams(location.search);
window.otrvRender = {
  ready: new Promise(r => setTimeout(r, 50)),
  duration: 1.0,
  async frame(t, layer, opts = {}) {   // viewer v2: frame(t, layer, {cam})
    if (p.get('render') !== '1') throw new Error('not in render mode');
    g.clearRect(0, 0, c.width, c.height);
    if (layer === 'splat') {            // red, green channel encodes t, blue encodes the camera
      g.fillStyle = `rgb(200, ${Math.round(t * 200)}, ${{driver: 255, drone: 128}[opts.cam] ?? 0})`;
      g.fillRect(0, 0, c.width, c.height);
    } else if (layer === 'overlay') {   // opaque blue square in the middle, transparent elsewhere
      g.fillStyle = 'rgb(0, 0, 255)';
      g.fillRect(c.width / 4, c.height / 4, c.width / 2, c.height / 2);
    }
  },
};
</script></body></html>"""


REQUIRE_BROWSER = bool(os.environ.get("OTRV_REQUIRE_RENDER_TEST"))  # CI: a missing browser is a failure
needs_browser = pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None and not REQUIRE_BROWSER,
    reason="playwright not installed (extra: render)",
)


def _render(work, page, opts):
    try:
        return render.run(work, page, opts)
    except Exception as e:  # browser binary not downloaded: `playwright install chromium`
        if not REQUIRE_BROWSER and ("Executable doesn't exist" in str(e) or "playwright install" in str(e)):
            pytest.skip("Chromium for Playwright not installed")
        raise


@needs_browser
def test_render_fails_fast_when_the_page_never_defines_the_api(work, monkeypatch):
    monkeypatch.setattr(render, "API_TIMEOUT_S", 1)
    page = b"<html><body><canvas></canvas><!-- otrvRender is mentioned but never defined --></body></html>"
    t0 = time.monotonic()
    with pytest.raises(SystemExit, match="did not expose window.otrvRender within 1 s"):
        _render(work, page, render.Options(out=work / "x.mp4", size=(160, 90), swiftshader=True))
    assert time.monotonic() - t0 < 20  # browser start + 1 s, not the old 30 s wait


@needs_browser
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_render_loop_with_fake_viewer(work, capsys):
    out = work / "render.mp4"
    opts = render.Options(
        out=out, fps=4, size=(160, 90), aspect="16:9", cam="driver", keep_frames=True, swiftshader=True
    )
    _render(work, FAKE_VIEWER, opts)
    splat = sorted((work / "render" / "splat").glob("*.png"))
    overlay = sorted((work / "render" / "overlay").glob("*.png"))
    assert [p.name for p in splat] == ["00000.png", "00001.png", "00002.png", "00003.png"]
    assert [p.name for p in overlay] == [p.name for p in splat]
    for i, p in enumerate(splat):
        im = Image.open(p)
        assert im.size == (160, 90) and im.mode == "RGB"
        r, g, b = im.getpixel((5, 5))
        assert r == 200 and abs(g - round(i * 0.25 * 200)) <= 1 and b == 255  # t and {cam} reached frame()
    ov = np.asarray(Image.open(overlay[0]))
    assert ov.shape == (90, 160, 4)
    assert ov[45, 80, 3] == 255 and tuple(ov[45, 80, :3]) == (0, 0, 255) and ov[2, 2, 3] == 0
    probe = _probe(out)
    assert (probe["width"], probe["height"]) == (160, 90) and int(probe["nb_read_frames"]) == 4

    # same settings again: every frame is already captured, nothing is re-rendered
    capsys.readouterr()
    opts.keep_frames = False
    render.run(work, FAKE_VIEWER, opts)
    assert "capturing 0/4 frames" in capsys.readouterr().out
    assert not (work / "render").exists() and out.exists()

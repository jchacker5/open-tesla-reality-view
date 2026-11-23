import shutil
import subprocess
from pathlib import Path

import pytest

VIEWER = Path(__file__).resolve().parents[1] / "viewer" / "index.html"

REQUIRED = (
    'id="record"',
    'id="map"',
    'id="mapLabel"',
    "preserveDrawingBuffer: true",
    "maxPixelRadius: 96",
    "maxStdDev: 2",
    "MediaRecorder",
    "alignSlam",
    "leaflet@1.9.4",
    "World_Dark_Gray_Base",
    "EARTH_M = 6378137",
)


@pytest.fixture(scope="module")
def viewer_html():
    return VIEWER.read_text(encoding="utf-8")


@pytest.mark.parametrize("needle", REQUIRED)
def test_viewer_contains(viewer_html, needle):
    assert needle in viewer_html


def _first_module_body(html):
    open_tag = '<script type="module">'
    start = html.find(open_tag)
    assert start != -1, 'no <script type="module">'
    body_start = start + len(open_tag)
    end = html.find("</script>", body_start)
    assert end != -1, "unterminated module script"
    return html[body_start:end]


def test_viewer_module_syntax(viewer_html, tmp_path):
    body = _first_module_body(viewer_html)
    node = shutil.which("node")
    if node is None:
        return
    script = tmp_path / "viewer-module.mjs"
    script.write_text(body, encoding="utf-8")
    # --check is syntax only and does not execute the browser imports.
    proc = subprocess.run([node, "--check", str(script)], check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout

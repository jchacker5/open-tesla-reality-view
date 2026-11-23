import io
import json
import zipfile

import pytest

from tsplat import car

META = {"uid": "u", "name": "Model", "author": "artist", "url": "https://x", "downloadable": True, "faces": 1}


def test_licence_rules():
    assert car.check_licence({**META, "license_slug": "by", "license": "CC Attribution"}, False) is None
    assert "non-commercial" in car.check_licence({**META, "license_slug": "by-nc", "license": "CC BY-NC"}, False)
    assert car.check_licence({**META, "license_slug": "by-nc", "license": "CC BY-NC"}, True) is None
    assert car.check_licence({**META, "license_slug": "by-nd", "license": "ND"}, True)
    assert car.check_licence({**META, "downloadable": False, "license_slug": "by"}, False)


def test_credit_line():
    assert car.credit_line({**META, "license": "CC Attribution"}) == '"Model" by artist (https://x), CC Attribution'


def _zip(files: dict) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, data in files.items():
            z.writestr(n, data)
    return b.getvalue()


def test_install_raw_glb_and_zipped(tmp_path):
    assert car.install_model(b"glTF....", tmp_path / "a").name == "car.glb"
    out = car.install_model(_zip({"model/scene.glb": b"glTF1"}), tmp_path / "b")
    assert out.read_bytes() == b"glTF1"
    out = car.install_model(
        _zip({"src/scene.gltf": b"{}", "src/scene.bin": b"x", "src/textures/a.png": b"p"}), tmp_path / "c"
    )
    assert out == tmp_path / "c" / "car" / "scene.gltf"
    assert (tmp_path / "c" / "car" / "textures" / "a.png").exists()


def test_zip_slip_refused(tmp_path):
    with pytest.raises(SystemExit):
        car.install_model(_zip({"a/scene.gltf": b"{}", "../../evil.txt": b"x"}), tmp_path / "d")


def test_fetch_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("SKETCHFAB_TOKEN", raising=False)
    monkeypatch.setattr(car, "metadata", lambda uid: {**META, "license_slug": "by", "license": "CC Attribution"})
    with pytest.raises(SystemExit, match="API token"):
        car.fetch("u", tmp_path, None)


def test_sync_into(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "car.glb").write_bytes(b"glTF")
    (models / "car.json").write_text(json.dumps({"file": "car.glb"}))
    monkeypatch.setenv("OTRV_MODELS", str(models))
    vd = tmp_path / "viewer"
    vd.mkdir()
    car.sync_into(vd)
    assert (vd / "models" / "car.glb").exists()
    car.clear(models)
    car.sync_into(vd)
    assert not (vd / "models").exists()

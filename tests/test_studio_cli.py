import builtins

import pytest

from tsplat import __main__ as cli


def test_studio_args(monkeypatch):
    monkeypatch.delenv("OTRV_DATA", raising=False)
    a = cli.build_parser().parse_args(["studio", "--no-browser", "--port", "9000", "--data", "/x"])
    assert (a.cmd, a.no_browser, a.port, a.data) == ("studio", True, 9000, "/x")
    assert cli.build_parser().parse_args(["studio"]).port == 8770


def test_studio_without_extra_says_how_to_install(monkeypatch):
    real = builtins.__import__

    def fake(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("fastapi") or (name == "studio" and level == 1 and "server" in (fromlist or ())):
            raise ImportError("No module named 'fastapi'", name="fastapi")  # as if the extra weren't installed
        return real(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(SystemExit) as e:
        cli.main(["studio", "--no-browser"])
    assert "uv sync --extra studio" in str(e.value.code)


def test_studio_serves(monkeypatch):
    pytest.importorskip("fastapi")
    from tsplat.studio import server

    seen = {}

    def fake_serve(data, port, open_browser, work):
        seen.update(data=data, port=port, open_browser=open_browser, work=work)

    monkeypatch.setattr(server, "serve", fake_serve)
    cli.main(["--data", "/footage", "studio", "--no-browser", "--port", "8800"])
    assert seen == {"data": "/footage", "port": 8800, "open_browser": False, "work": "work"}

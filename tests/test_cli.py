import argparse
import json
import socket
import subprocess

import pytest

from tsplat import __main__ as cli


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OTRV_DATA", raising=False)
    monkeypatch.setenv("OTRV_SKIP_DOCTOR", "1")
    return tmp_path


def run(*argv):
    with pytest.raises(SystemExit) as e:
        cli.main(list(argv))
    return str(e.value.code)


def write_index(folder, clips):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.json").write_text(json.dumps(clips))


CLIPS = [
    {"id": "2026-01-02_03-04-05", "timestamp": "2026-01-02_03-04-05", "kind": "RecentClips", "seconds": 60.0,
     "cams": dict.fromkeys(["front", "back", "left_repeater", "right_repeater"], "/x"), "event": None,
     "mph": 24.6, "mph_1hz": [25.0] * 60},
    {"id": "ev/2026-01-03_00-00-00", "timestamp": "2026-01-03_00-00-00", "kind": "SentryClips", "seconds": 60.0,
     "cams": dict.fromkeys(["front", "back", "left_repeater", "right_repeater"], "/x"),
     "event": {"reason": "sentry_aware_object_detection"}, "mph": None, "mph_1hz": None},
]  # fmt: skip


@pytest.mark.parametrize(
    ("argv", "hint"),
    [
        (["list"], "run `otrv ingest` first"),
        (["egomask"], "run `otrv ingest` first"),
        (["suggest"], "run `otrv ingest` first"),
        (["prep", "x", "--name", "a"], "run `otrv ingest` first"),
        (["view", "a"], "run `otrv prep <clip> --name a` first"),
        (["bench", "a"], "run `otrv prep <clip> --name a` first"),
        (["nvs-export", "a"], "run `otrv prep <clip> --name a` first"),
        (["recon", "a"], "run `otrv prep <clip> --name a` first"),
    ],
)
def test_stage_guards(argv, hint):
    assert hint in run(*argv)


@pytest.mark.parametrize("argv", [["view", "a"], ["bench", "a"], ["nvs-export", "a"]])
def test_guards_after_prep_point_to_recon(in_tmp, argv):
    (in_tmp / "work" / "a").mkdir(parents=True)
    (in_tmp / "work" / "a" / "meta.json").write_text("{}")
    assert "run `otrv recon a` first" in run(*argv)


def test_list_shows_header_and_speed(in_tmp, capsys):
    write_index(in_tmp / "data" / "teslacam", CLIPS)
    cli.main(["list"])
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == ["CLIP", "KIND", "LENGTH", "CAMS", "MPH", "EVENT"]
    assert out[1].split()[:5] == ["2026-01-02_03-04-05", "RecentClips", "60s", "4", "25"]
    assert out[2].split()[4] == "-"  # no SEI


@pytest.mark.parametrize("where", ["env", "before", "after"])
def test_data_dir_option_and_env(in_tmp, monkeypatch, capsys, where):
    data = in_tmp / "elsewhere"
    write_index(data, CLIPS)
    if where == "env":
        monkeypatch.setenv("OTRV_DATA", str(data))
        cli.main(["list"])
    elif where == "before":
        cli.main(["--data", str(data), "list"])
    else:
        cli.main(["list", "--data", str(data)])
    assert "2026-01-02_03-04-05" in capsys.readouterr().out


def test_suggest_prints_ready_to_paste_prep(in_tmp, capsys):
    data = in_tmp / "d"
    write_index(data, CLIPS)
    cli.main(["suggest", "--data", str(data)])
    out = capsys.readouterr().out
    assert f"otrv --data {data} prep 2026-01-02_03-04-05 --name drive_2026-01-02_03-04-05 --start 0 --dur 20" in out
    assert "ev/" not in out  # Sentry excluded


def test_ingest_kinds_default_is_plain_text_in_help():
    parser = cli.build_parser()
    ingest_p = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices["ingest"]
    help_text = " ".join(ingest_p.format_help().split())
    assert "(default: RecentClips,SavedClips)" in help_text and "('RecentClips'" not in help_text
    assert parser.parse_args(["ingest"]).kinds == ("RecentClips", "SavedClips")
    assert parser.parse_args(["ingest", "--kinds", "SentryClips"]).kinds == ("SentryClips",)


def test_ingest_dest_is_an_alias_for_data(in_tmp, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.ingest, "run", lambda src, data, *a: seen.update(data=data))
    cli.main(["ingest", "--index-only", "--dest", "old/place"])
    assert seen["data"] == "old/place"


def test_called_process_error_is_one_line(monkeypatch):
    def boom(args):
        raise subprocess.CalledProcessError(3, ["/opt/homebrew/bin/colmap", "mapper", "--database_path", "x"])

    monkeypatch.setattr(cli, "dispatch", boom)
    msg = run("doctor")
    assert "`colmap mapper` failed (exit 3)" in msg and "--debug" in msg and "\n" not in msg
    with pytest.raises(subprocess.CalledProcessError):
        cli.main(["--debug", "doctor"])
    with pytest.raises(subprocess.CalledProcessError):
        cli.main(["doctor", "--debug"])


def test_file_not_found_is_one_line(monkeypatch):
    def boom(args):
        raise FileNotFoundError(2, "No such file or directory", "ffprobe")

    monkeypatch.setattr(cli, "dispatch", boom)
    msg = run("doctor")
    assert "not found: ffprobe" in msg and "otrv doctor" in msg


def test_viewer_html_prefers_packaged_copy(tmp_path, monkeypatch):
    # source checkout / editable install: no packaged copy, so the repo's viewer/ is used
    assert cli.viewer_html() == cli.ROOT / "viewer" / "index.html"
    pkg = tmp_path / "tsplat"
    (pkg / "viewer").mkdir(parents=True)
    (pkg / "viewer" / "index.html").write_text("<html>packaged</html>")
    monkeypatch.setattr(cli.resources, "files", lambda name: pkg)  # wheel install
    assert cli.viewer_html().read_text() == "<html>packaged</html>"


def test_view_port_in_use_suggests_another(in_tmp):
    vd = in_tmp / "work" / "a" / "viewer"
    vd.mkdir(parents=True)
    (in_tmp / "work" / "a" / "meta.json").write_text("{}")
    (vd / "scene.ply").write_text("")
    (vd / "path.json").write_text("{}")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        msg = run("view", "a", "--port", str(port))
    assert f"port {port} is already in use" in msg and f"--port {port + 1}" in msg


def _all_actions(parser):
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            for name, p in a.choices.items():
                for b in p._actions:
                    yield name, b
        else:
            yield "otrv", a


def test_every_option_has_help_and_prog_is_otrv():
    parser = cli.build_parser()
    assert parser.prog == "otrv"
    for cmd, action in _all_actions(parser):
        if isinstance(action, argparse._SubParsersAction) or action.help == argparse.SUPPRESS:
            continue
        assert action.help, f"{cmd} {action.option_strings or action.dest} has no help"
    text = parser.format_help() + "".join(
        p.format_help()
        for a in parser._actions
        if isinstance(a, argparse._SubParsersAction)
        for p in a.choices.values()
    )
    assert "6 cameras" not in text and "tsplat" not in text
    assert "(default: 15000)" in " ".join(text.split())  # ArgumentDefaultsHelpFormatter (may wrap)

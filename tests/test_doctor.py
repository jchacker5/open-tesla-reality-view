import subprocess

import pytest

from tsplat import brush, doctor

COLMAP_4 = "COLMAP 4.2.1 (Commit abc)\nUsage:\n  colmap [command]\nAvailable commands:\n  mapper\n  rig_configurator\n"
COLMAP_3 = "COLMAP 3.12.0 (Commit abc)\nAvailable commands:\n  mapper\n  rig_configurator\n"


def set_free_gb(monkeypatch, gb):
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: type("U", (), {"free": gb * 2**30})())


@pytest.fixture
def tools(monkeypatch, tmp_path):
    """Fake PATH + subprocess: `found` maps tool -> path, `outputs` maps argv[0] name -> stdout."""
    found = {"ffmpeg": "/bin/ffmpeg", "ffprobe": "/bin/ffprobe", "colmap": "/bin/colmap"}
    outputs = {"colmap": COLMAP_4, "brush_app": "brush-cli 0.3.0\n"}
    exe = tmp_path / "brush_app"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("BRUSH", str(exe))
    monkeypatch.delenv("OTRV_SKIP_DOCTOR", raising=False)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: found.get(name))
    set_free_gb(monkeypatch, 500)

    def fake_run(cmd, **kw):
        name = cmd[0].rsplit("/", 1)[-1]
        if name not in outputs:
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, 0, outputs[name], "")

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    return found, outputs, exe


def by_name(checks):
    return {c.name: c for c in checks}


def test_all_good(tools, monkeypatch):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(doctor.platform, "machine", lambda: "arm64")
    c = by_name(doctor.run_checks(["platform", "ffmpeg", "ffprobe", "colmap", "brush"]))
    assert all(x.ok for x in c.values()), c
    assert c["colmap"].detail.startswith("4.2") and c["brush"].detail.startswith("0.3.0")


def test_missing_ffmpeg_prints_install_command(tools, monkeypatch):
    found, _, _ = tools
    del found["ffprobe"]
    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    c = doctor.check_tool("ffprobe", "ffmpeg")
    assert not c.ok and c.fix == "sudo apt install ffmpeg"
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    assert doctor.check_tool("ffprobe", "ffmpeg").fix == "brew install ffmpeg"


def test_old_colmap_fails_with_upgrade_fix(tools, monkeypatch):
    _, outputs, _ = tools
    outputs["colmap"] = COLMAP_3
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    c = doctor.check_colmap()
    assert not c.ok and "3.12" in c.detail and "need 4.0+" in c.detail and "brew" in c.fix


def test_colmap_without_rig_configurator_fails(tools):
    _, outputs, _ = tools
    outputs["colmap"] = "COLMAP 4.0.0\nAvailable commands:\n  mapper\n"
    c = doctor.check_colmap()
    assert not c.ok and "rig_configurator" in c.detail


def test_brush_missing_not_executable_or_wrong_version(tools, monkeypatch, tmp_path):
    _, outputs, exe = tools
    outputs["brush_app"] = "brush-cli 0.2.1\n"
    c = doctor.check_brush()
    assert not c.ok and "0.3.x" in c.detail and "otrv setup-brush" in c.fix

    exe.chmod(0o644)
    c = doctor.check_brush()
    assert not c.ok and c.fix.startswith("chmod +x")

    monkeypatch.setenv("BRUSH", str(tmp_path / "nope"))
    c = doctor.check_brush()
    assert not c.ok and "not found" in c.detail and "otrv setup-brush" in c.fix


def test_brush_that_cannot_run_suggests_quarantine_fix_on_mac(tools, monkeypatch):
    _, outputs, _ = tools
    del outputs["brush_app"]  # running it raises OSError
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    c = doctor.check_brush()
    assert not c.ok and "com.apple.quarantine" in c.fix


def test_intel_mac_is_reported_clearly(monkeypatch):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(doctor.platform, "machine", lambda: "x86_64")
    c = doctor.check_platform()
    assert not c.ok and "Intel" in c.fix


def test_disk_and_ram(monkeypatch, tmp_path):
    set_free_gb(monkeypatch, 3)
    c = doctor.check_disk(tmp_path / "not" / "yet")  # walks up to an existing parent
    assert not c.ok and c.warn_only and "3.0 GB" in c.detail and "10 GB+ recommended" in c.detail
    set_free_gb(monkeypatch, 1.5)
    c = doctor.check_disk(tmp_path)
    assert not c.ok and not c.warn_only and "need 2 GB+" in c.detail
    set_free_gb(monkeypatch, 50)
    assert doctor.check_disk(tmp_path).ok
    monkeypatch.setattr(doctor, "ram_gb", lambda: 8.0)
    c = doctor.check_ram()
    assert not c.ok and c.warn_only and "--steps" in c.fix


def test_preflight_fails_early_with_fix(tools, monkeypatch):
    found, _, _ = tools
    del found["ffmpeg"]
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    with pytest.raises(SystemExit) as e:
        doctor.preflight("prep")
    msg = str(e.value)
    assert "otrv prep" in msg and "ffmpeg" in msg and "brew install ffmpeg" in msg and "ffprobe" not in msg
    doctor.preflight("prep", skip=("ffmpeg",))  # skipped -> passes
    monkeypatch.setenv("OTRV_SKIP_DOCTOR", "1")
    doctor.preflight("prep")


@pytest.mark.parametrize("step", ["prep", "recon"])
def test_preflight_checks_disk_for_prep_and_recon(tools, monkeypatch, capsys, step):
    set_free_gb(monkeypatch, 5)
    doctor.preflight(step)  # low: warns, still runs
    assert "warning before `otrv " + step in capsys.readouterr().out
    set_free_gb(monkeypatch, 1)
    with pytest.raises(SystemExit, match="need 2 GB"):
        doctor.preflight(step)


def test_preflight_recon_checks_colmap_and_brush(tools, monkeypatch):
    _, outputs, _ = tools
    outputs["colmap"] = COLMAP_3
    with pytest.raises(SystemExit, match="colmap"):
        doctor.preflight("recon")
    doctor.preflight("recon", skip=("colmap",))  # --skip-sfm doesn't need COLMAP


def test_doctor_exit_code(tools, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "CHECKS", {"ok": lambda: doctor.Check("ok", True, "fine")})
    assert doctor.doctor() == 0
    monkeypatch.setattr(
        doctor,
        "CHECKS",
        {
            "warn": lambda: doctor.Check("warn", False, "meh", "x", warn_only=True),
            "bad": lambda: doctor.Check("bad", False, "broken", "do this"),
        },
    )
    assert doctor.doctor() == 1
    out = capsys.readouterr().out
    assert "FAIL  bad" in out and "fix: do this" in out and "warn  warn" in out


def test_brush_target_triples():
    assert brush.target("Darwin", "arm64") == "aarch64-apple-darwin"
    assert brush.target("Linux", "x86_64") == "x86_64-unknown-linux-gnu"
    assert brush.target("Windows", "AMD64") == "x86_64-pc-windows-msvc"
    assert brush.target("Darwin", "x86_64") is None
    assert brush.target("Linux", "aarch64") is None

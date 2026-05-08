from __future__ import annotations

import contextlib
import subprocess


def test_uv_pip_install_uses_uv_environment_variables(monkeypatch):
    from app.ui.launcher import core

    monkeypatch.setattr(
        core,
        "PATHS",
        {
            "UV_EXE": "uv.exe",
            "REQ_FILE": "requirements_cu13.txt",
            "PYTHON_EXE": "python.exe",
            "APP_DIR": "repo",
        },
    )
    monkeypatch.setenv("UV_HTTP_TIMEOUT", "999")
    monkeypatch.delenv("UV_HTTP_RETRIES", raising=False)
    monkeypatch.delenv("UV_CONCURRENT_DOWNLOADS", raising=False)

    recorded = {}

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(core.subprocess, "run", fake_run)

    result = core.uv_pip_install()

    assert result.returncode == 0
    assert recorded["cmd"] == [
        "uv.exe",
        "pip",
        "install",
        "-r",
        "requirements_cu13.txt",
        "--python",
        "python.exe",
    ]
    assert "--timeout" not in recorded["cmd"]
    assert "--retries" not in recorded["cmd"]
    assert "--concurrent-downloads" not in recorded["cmd"]
    assert recorded["kwargs"]["check"] is True
    assert recorded["kwargs"]["env"]["UV_HTTP_TIMEOUT"] == "999"
    assert recorded["kwargs"]["env"]["UV_HTTP_RETRIES"] == "5"
    assert recorded["kwargs"]["env"]["UV_CONCURRENT_DOWNLOADS"] == "4"


def test_update_deps_failure_skips_checksum_update(monkeypatch):
    from app.ui.launcher import launcher_window

    checksum_writes = []
    dialogs = []

    monkeypatch.setattr(
        launcher_window,
        "with_busy_state",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        launcher_window,
        "uv_pip_install",
        lambda: (_ for _ in ()).throw(
            subprocess.CalledProcessError(2, ["uv", "pip", "install"])
        ),
    )
    monkeypatch.setattr(
        launcher_window,
        "write_checksum_state",
        lambda **kwargs: checksum_writes.append(kwargs),
    )
    monkeypatch.setattr(
        launcher_window.QtWidgets.QMessageBox,
        "critical",
        lambda *args: dialogs.append(args),
    )

    launcher_window.LauncherWindow.on_update_deps(object())

    assert checksum_writes == []
    assert dialogs


def test_update_deps_success_updates_checksum_and_refreshes(monkeypatch):
    from app.ui.launcher import launcher_window

    events = []

    class DummyWindow:
        def _load_checksum_status(self):
            events.append("load")

        def _refresh_update_indicators(self):
            events.append("refresh")

    monkeypatch.setattr(
        launcher_window,
        "with_busy_state",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(launcher_window, "uv_pip_install", lambda: events.append("uv"))
    monkeypatch.setattr(
        launcher_window, "compute_file_sha256", lambda _path: "deps-sha"
    )
    monkeypatch.setattr(
        launcher_window,
        "write_checksum_state",
        lambda **kwargs: events.append(("checksum", kwargs)),
    )

    launcher_window.LauncherWindow.on_update_deps(DummyWindow())

    assert events == [
        "uv",
        ("checksum", {"deps_sha": "deps-sha"}),
        "load",
        "refresh",
    ]

"""Console streams must survive log lines with non-ANSI-codepage characters."""

from __future__ import annotations

import io
import sys
from pathlib import Path

from app.helpers.console_color import install_utf8_console_streams


def test_install_utf8_console_streams_reconfigures_redirected_output():
    """Redirected stdout defaults to the ANSI code page, where '≤' raises."""
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252", line_buffering=True)
    original = sys.stdout
    sys.stdout = stream
    try:
        install_utf8_console_streams()
        print("[INFO] Display: Wall-clock catch-up (target≤900, had been at 880)")
        sys.stdout.flush()
    finally:
        sys.stdout = original

    assert stream.encoding == "utf-8"
    assert "target≤900" in buffer.getvalue().decode("utf-8")


def test_install_utf8_console_streams_tolerates_streams_without_reconfigure():
    original = sys.stdout
    sys.stdout = io.StringIO()
    try:
        install_utf8_console_streams()
    finally:
        sys.stdout = original


def test_main_installs_utf8_streams_before_colors():
    body = (
        Path("main.py")
        .read_text(encoding="utf-8")
        .split('if __name__ == "__main__":')[1]
    )
    assert body.index("install_utf8_console_streams()") < body.index(
        "install_colored_console_streams()"
    )

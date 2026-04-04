"""
ANSI colors for console log lines ([INFO], [WARN], [ERROR]).

Enabled only for interactive TTY output; respects NO_COLOR. On Windows, enables
virtual terminal processing when possible.
"""

from __future__ import annotations

import os
import sys
from typing import Any, TextIO

_RESET = "\033[0m"
_WHITE = "\033[97m"  # bright white (readable on dark consoles)
_YELLOW = "\033[33m"
_RED = "\033[31m"


def _try_enable_windows_vt_mode() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        enable_vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        for std_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(std_id)
            if handle in (-1, 0):
                continue
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
                continue
            kernel32.SetConsoleMode(handle, mode.value | enable_vt)
    except OSError:
        pass


def _line_prefix_tag(line: str) -> str | None:
    """Return tag if line starts with [INFO]/[WARN]/[ERROR] (after \\r)."""
    s = line.lstrip("\r")
    if s.startswith("[ERROR]"):
        return "error"
    if s.startswith("[WARN]"):
        return "warn"
    if s.startswith("[INFO]"):
        return "info"
    return None


def _colorize_complete_line(line: str) -> str:
    tag = _line_prefix_tag(line)
    if tag == "error":
        return f"{_RED}{line}{_RESET}"
    if tag == "warn":
        return f"{_YELLOW}{line}{_RESET}"
    if tag == "info":
        return f"{_WHITE}{line}{_RESET}"
    return line


class _AnsiLogColorStream:
    """Buffer by newline; colorize lines that start with [INFO]/[WARN]/[ERROR]."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._buffer = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        out: list[str] = []
        while True:
            n = self._buffer.find("\n")
            if n == -1:
                break
            line = self._buffer[: n + 1]
            self._buffer = self._buffer[n + 1 :]
            out.append(_colorize_complete_line(line))
        if out:
            self._stream.write("".join(out))
        return len(s)

    def flush(self) -> None:
        if self._buffer:
            self._stream.write(_colorize_complete_line(self._buffer))
            self._buffer = ""
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    @property
    def encoding(self) -> str:
        enc = getattr(self._stream, "encoding", None)
        return enc if isinstance(enc, str) else "utf-8"

    def fileno(self) -> int:
        return self._stream.fileno()


def install_colored_console_streams() -> None:
    """
    Wrap sys.stdout and sys.stderr when output is a TTY and colors are allowed.

    Call once at process startup (before other threads print heavily).
    """
    if os.environ.get("NO_COLOR", "").strip():
        return
    if os.environ.get("TERM", "").lower() == "dumb":
        return

    _try_enable_windows_vt_mode()

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        if not getattr(stream, "isatty", lambda: False)():
            continue
        if isinstance(stream, _AnsiLogColorStream):
            continue
        setattr(sys, name, _AnsiLogColorStream(stream))

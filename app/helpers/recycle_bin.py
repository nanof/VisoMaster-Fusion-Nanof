"""Move files to the OS recycle bin / trash with a Windows fallback."""

from __future__ import annotations

import os
import sys
from typing import Union

StrPath = Union[str, os.PathLike[str]]


def _normalize_path(path: StrPath) -> str:
    return os.path.normpath(os.path.abspath(os.fspath(path)))


def _windows_shfileop_recycle(path: str) -> None:
    """Use SHFileOperationW with FOF_ALLOWUNDO; no 8.3 short-path step."""
    from ctypes import (
        FormatError,
        Structure,
        addressof,
        byref,
        c_uint,
        create_unicode_buffer,
        windll,
    )
    from ctypes.wintypes import BOOL, HWND, LPCWSTR, UINT

    from send2trash.win.legacy import convert_sh_file_opt_result

    shell32 = windll.shell32
    SHFileOperationW = shell32.SHFileOperationW

    class SHFILEOPSTRUCTW(Structure):
        _fields_ = [
            ("hwnd", HWND),
            ("wFunc", UINT),
            ("pFrom", LPCWSTR),
            ("pTo", LPCWSTR),
            ("fFlags", c_uint),
            ("fAnyOperationsAborted", BOOL),
            ("hNameMappings", c_uint),
            ("lpszProgressTitle", LPCWSTR),
        ]

    FO_DELETE = 3
    FOF_SILENT = 4
    FOF_NOCONFIRMATION = 16
    FOF_ALLOWUNDO = 64
    FOF_NOERRORUI = 1024

    paths = [path]
    buffer = create_unicode_buffer(" ".join(paths))
    path_string = "\0".join(paths)
    buffer = create_unicode_buffer(path_string, len(buffer) + 1)
    fileop = SHFILEOPSTRUCTW()
    fileop.hwnd = 0
    fileop.wFunc = FO_DELETE
    fileop.pFrom = LPCWSTR(addressof(buffer))
    fileop.pTo = None
    fileop.fFlags = (
        FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT
    )
    fileop.fAnyOperationsAborted = 0
    fileop.hNameMappings = 0
    fileop.lpszProgressTitle = None
    result = SHFileOperationW(byref(fileop))
    if result:
        error = convert_sh_file_opt_result(result)
        raise OSError(error, FormatError(error), path)


def recycle_path(path: StrPath) -> None:
    """
    Move *path* to the recycle bin / trash.

    send2trash's Windows legacy backend calls GetShortPathNameW, which fails when
    NTFS 8.3 short names are disabled. We normalize paths, try send2trash first,
    then fall back to SHFileOperationW without the short-path conversion.
    """
    p = _normalize_path(path)
    from send2trash import send2trash

    try:
        send2trash(p)
    except OSError:
        if sys.platform != "win32":
            raise
        _windows_shfileop_recycle(p)

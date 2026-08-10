"""
Ensure embeddable/portable CPython has Include/Python.h + libs/pythonXY.lib.

Triton / torch.compile on Windows JIT-links a small host extension against the
interpreter. Official embeddable builds omit those files; this script copies
them from the matching NuGet ``python`` package into ``sys.base_exec_prefix``.

Usage (from repo or portable root)::

    portable-files\\python\\python.exe scripts/ensure_portable_python_dev_headers.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def _already_present(root: Path) -> bool:
    vi = sys.version_info
    nodot = f"{vi.major}{vi.minor}"
    return (root / "Include" / "Python.h").is_file() and (
        root / "libs" / f"python{nodot}.lib"
    ).is_file()


def ensure_dev_headers(root: Path | None = None) -> bool:
    """Return True when headers/libs are present (already or after install)."""
    root = Path(root or sys.base_exec_prefix)
    if _already_present(root):
        print(f"[OK] Python dev files already present under {root}")
        return True

    vi = sys.version_info
    version = f"{vi.major}.{vi.minor}.{vi.micro}"
    url = f"https://www.nuget.org/api/v2/package/python/{version}"
    print(f"[INFO] Fetching NuGet python/{version} for Include/ + libs/ ...")
    with tempfile.TemporaryDirectory(prefix="py_nuget_") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "python.nupkg"
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            return False
        extract = tmp_path / "extract"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract)
        include_src = extract / "tools" / "include"
        libs_src = extract / "tools" / "libs"
        if not include_src.is_dir() or not libs_src.is_dir():
            print(f"[ERROR] Unexpected NuGet layout under {extract / 'tools'}")
            return False
        shutil.copytree(include_src, root / "Include", dirs_exist_ok=True)
        shutil.copytree(libs_src, root / "libs", dirs_exist_ok=True)

    if _already_present(root):
        print(f"[OK] Installed Include/ and libs/ into {root}")
        return True
    print("[ERROR] Copy finished but Python.h / pythonXY.lib still missing")
    return False


def main() -> int:
    return 0 if ensure_dev_headers() else 1


if __name__ == "__main__":
    raise SystemExit(main())

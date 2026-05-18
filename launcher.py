#!/usr/bin/env python3
"""Launcher shim that works regardless of the caller's working directory.

Use this script as the entry point for the VisoMaster Fusion launcher when the
caller cannot guarantee the cwd is the repo root:

    python launcher.py

This is the **path-based** equivalent of:

    python -m app.ui.launcher

The `-m` form fails with `ModuleNotFoundError: No module named 'app'` whenever
Python is invoked from outside the repo root, because Python's `runpy` resolves
the module before executing any user code (so `sys.path.insert` inside
`app/ui/launcher/__main__.py` is too late to fix the import).

This shim resolves the repo root from its own `__file__`, prepends it to
`sys.path`, and then performs a normal import — guaranteed to succeed regardless
of cwd. Portable installers and post-install steps should call this file by
absolute path rather than the `-m` form.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.ui.launcher.main import main  # noqa: E402

if __name__ == "__main__":
    main()

# __main__.py
# ---------------------------------------------------------------------------
# VisoMaster Fusion Launcher Entrypoint
# ---------------------------------------------------------------------------
# Runs the VisoMaster Fusion Launcher GUI via:
#   python -m app.ui.launcher
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

# Ensure the repository root (the directory containing the `app/` package) is on
# sys.path regardless of the working directory the interpreter was launched from.
# This prevents "ModuleNotFoundError: No module named 'app'" when the portable
# install script invokes `python -m app.ui.launcher` from the wrong directory.
_repo_root = (
    Path(__file__).resolve().parent.parent.parent.parent
)  # .../VisoMaster-Fusion
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from .main import main

    print("[Launcher] Starting VisoMaster Fusion Launcher...")
    main()
except Exception as e:
    print(f"[Launcher] Failed to start the VisoMaster Fusion Launcher: {e}")
    sys.exit(1)

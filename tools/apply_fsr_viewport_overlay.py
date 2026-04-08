"""Apply FSR viewport-overlay refactor to video_preview_fsr_gl_item.py (run once from repo root)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "app/ui/widgets/video_preview_fsr_gl_item.py"
helpers = (ROOT / "tools/fsr_viewport_helpers.txt").read_text(encoding="utf-8")
body = (ROOT / "tools/fsr_viewport_body.txt").read_text(encoding="utf-8")
text = path.read_text(encoding="utf-8")
i = text.find("    def paint(  # noqa: N802")
j = text.find("    def reset_gl_state(self)", i)
if i < 0 or j < 0:
    raise SystemExit("markers not found")
path.write_text(text[:i] + helpers + body + text[j:], encoding="utf-8")
print("patched", path)

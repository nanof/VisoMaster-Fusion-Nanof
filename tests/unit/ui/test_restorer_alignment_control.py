from __future__ import annotations

from pathlib import Path


def test_face_restorer_alignment_layout_has_no_restorer_load_hook():
    text = Path("app/ui/widgets/common_layout_data.py").read_text(encoding="utf-8")
    start = text.index('"FaceRestorerDetTypeSelection"')
    end = text.index('"FaceFidelityWeightDecimalSlider"', start)
    block = text[start:end]
    assert "exec_function" not in block
    assert "FaceRestorerEnable2Toggle" not in block

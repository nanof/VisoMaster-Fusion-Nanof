"""
Stub para retalking / audio-driven head (p. ej. Wav2Lip, SadTalker, edtalk ONNX/PyTorch).

Pipeline típico: audio + vídeo → cola offline (no hot path de swap por frame).
"""

from __future__ import annotations

from app.processors.pytorch_extras import is_pytorch_extras_enabled


def run_retalking_placeholder(*_args, **_kwargs) -> None:
    if not is_pytorch_extras_enabled():
        print(
            "[INFO] Retalking: defina VISOFUSION_PYTORCH_EXTRAS=1 e instale "
            "requirements-pytorch-extra.txt para habilitar este pipeline."
        )
        return
    print(
        "[INFO] Retalking: stub — añadir orquestación FFmpeg + modelo "
        "(p. ej. edtalk_256.onnx en pytorch_weights/) en una iteración posterior."
    )

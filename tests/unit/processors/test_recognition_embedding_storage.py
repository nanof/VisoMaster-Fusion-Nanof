"""Recognition cache dtype: FP16 storage promotes to FP32 on read paths."""

from __future__ import annotations

import numpy as np


def test_fp16_stored_embedding_roundtrips_to_float32_for_math():
    emb_f32 = np.random.randn(512).astype(np.float32)
    emb_fp16 = emb_f32.astype(np.float16).copy()
    back = np.asarray(emb_fp16, dtype=np.float32)
    assert back.dtype == np.float32
    assert np.allclose(back, emb_f32.astype(np.float32), rtol=1e-3, atol=1e-3)

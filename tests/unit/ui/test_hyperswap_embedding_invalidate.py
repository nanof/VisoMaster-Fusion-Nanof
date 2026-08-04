"""Invalidate HyperSwapArcFace caches when SwapModelSelection changes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from app.helpers.hyperswap_embedding import (
    HYPERSWAP_ARCFACE_KEY,
    invalidate_hyperswap_arcface_embeddings,
    pop_hyperswap_arcface_from_store,
)


def test_pop_hyperswap_arcface_from_store():
    store = {"Inswapper128ArcFace": np.ones(4), HYPERSWAP_ARCFACE_KEY: np.zeros(4)}
    assert pop_hyperswap_arcface_from_store(store) is True
    assert HYPERSWAP_ARCFACE_KEY not in store
    assert "Inswapper128ArcFace" in store
    assert pop_hyperswap_arcface_from_store(store) is False
    assert pop_hyperswap_arcface_from_store(None) is False


def test_invalidate_hyperswap_arcface_clears_stores_and_recalculates():
    emb = np.ones(512, dtype=np.float32)
    input_store = {"Inswapper128ArcFace": emb.copy(), HYPERSWAP_ARCFACE_KEY: emb.copy()}
    input_face = SimpleNamespace(embedding_store=input_store, face_id="in1")

    assigned_store = input_store  # shared reference, like assign path
    target = SimpleNamespace(
        assigned_input_faces={"in1": assigned_store},
        assigned_merged_embeddings={},
        assigned_input_embedding={HYPERSWAP_ARCFACE_KEY: emb.copy()},
        aged_input_embedding={HYPERSWAP_ARCFACE_KEY: emb.copy()},
        calculate_assigned_input_embedding=MagicMock(),
    )

    main_window = SimpleNamespace(
        input_faces={"in1": input_face},
        merged_embeddings={},
        target_faces={"t1": target},
    )

    cleared = invalidate_hyperswap_arcface_embeddings(main_window)
    assert cleared >= 1
    assert HYPERSWAP_ARCFACE_KEY not in input_store
    assert "Inswapper128ArcFace" in input_store
    assert HYPERSWAP_ARCFACE_KEY not in target.assigned_input_embedding
    assert HYPERSWAP_ARCFACE_KEY not in target.aged_input_embedding
    target.calculate_assigned_input_embedding.assert_called_once()


def test_swap_model_selection_wires_invalidate_handler():
    text = Path("app/ui/widgets/swapper_layout_data.py").read_text(encoding="utf-8")
    assert "on_swap_model_selection_change" in text
    assert "HyperSwapArcFace" in text

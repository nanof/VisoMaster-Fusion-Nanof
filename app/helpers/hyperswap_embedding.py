"""HyperSwapArcFace cache invalidation (no Qt / layout imports)."""

from __future__ import annotations

from typing import Any

HYPERSWAP_ARCFACE_KEY = "HyperSwapArcFace"
HYPERSWAP_SWAPPER_MODELS = frozenset(
    {"HyperSwap-v1", "HyperSwap-v2", "HyperSwap-v3"}
)


def pop_hyperswap_arcface_from_store(store: Any) -> bool:
    """Remove ``HyperSwapArcFace`` from an embedding dict if present."""
    if not isinstance(store, dict) or HYPERSWAP_ARCFACE_KEY not in store:
        return False
    store.pop(HYPERSWAP_ARCFACE_KEY, None)
    return True


def invalidate_hyperswap_arcface_embeddings(main_window: Any) -> int:
    """Drop cached HyperSwapArcFace vectors so the next swap recomputes them.

    Input-face ``embedding_store`` dicts are shared by reference with
    ``assigned_input_faces``, so clearing the input cards covers assignments.
    Target ``assigned_input_embedding`` / aged stores are rebuilt afterward when
    ``calculate_assigned_input_embedding`` exists.
    """
    cleared = 0
    input_faces = getattr(main_window, "input_faces", None) or {}
    for ib in list(input_faces.values()):
        store = getattr(ib, "embedding_store", None)
        if pop_hyperswap_arcface_from_store(store):
            cleared += 1

    merged = getattr(main_window, "merged_embeddings", None) or {}
    for mb in list(merged.values()):
        store = getattr(mb, "embedding_store", None)
        if pop_hyperswap_arcface_from_store(store):
            cleared += 1

    target_faces = getattr(main_window, "target_faces", None) or {}
    for tb in list(target_faces.values()):
        if pop_hyperswap_arcface_from_store(
            getattr(tb, "assigned_input_embedding", None)
        ):
            cleared += 1
        if pop_hyperswap_arcface_from_store(getattr(tb, "aged_input_embedding", None)):
            cleared += 1
        # Some restore paths may hold distinct store copies — clear those too.
        assigned_faces = getattr(tb, "assigned_input_faces", None) or {}
        for store in list(assigned_faces.values()):
            pop_hyperswap_arcface_from_store(store)
        assigned_merged = getattr(tb, "assigned_merged_embeddings", None) or {}
        for store in list(assigned_merged.values()):
            pop_hyperswap_arcface_from_store(store)
        calc = getattr(tb, "calculate_assigned_input_embedding", None)
        if callable(calc):
            try:
                calc()
            except Exception as e:
                print(
                    f"[WARN] invalidate_hyperswap_arcface: recalculate failed: {e}",
                    flush=True,
                )
    return cleared

"""
Tests for the HyperSwap native mask blend.

The HyperSwap ONNX exports a second output ("mask") with the region the generator
considers valid. These tests cover binding it, degrading gracefully when the export or
engine rejects it, and folding it into ``swap_mask``.

All model inference is mocked.
"""

from __future__ import annotations

import types

import pytest
import torch


def _make_model(output_names: list[str], input_names: tuple[str, ...] = ("target",)):
    class _Model:
        def get_inputs(self):
            return [type("I", (), {"name": n})() for n in input_names]

        def get_outputs(self):
            return [type("O", (), {"name": n})() for n in output_names]

        def io_binding(self):
            return object()

    return _Model()


class _RecordingProc:
    """Minimal ModelsProcessor stand-in that records bound output names."""

    def __init__(self, fail_on_output: str | None = None):
        self.bound_outputs: list[str] = []
        self.fail_on_output = fail_on_output

    def bind_ort_io_input(self, _io, _model_name, _name, tensor, session=None):
        return tensor

    def bind_ort_io_output(self, _io, _model_name, name, _tensor, session=None):
        if self.fail_on_output is not None and name == self.fail_on_output:
            raise RuntimeError(f"cannot bind {name}")
        self.bound_outputs.append(name)


def _make_swappers(proc, model):
    from app.processors.face_swappers import FaceSwappers

    fs = FaceSwappers(models_processor=proc)  # type: ignore[arg-type]
    fs._load_swapper_model = lambda _n: model  # type: ignore[method-assign]
    fs.runs = 0

    def _run(*_a, **_k):
        fs.runs += 1

    fs._run_model_with_lazy_build_check = _run  # type: ignore[method-assign]
    return fs


# ---------------------------------------------------------------------------
# face_swappers: binding the second output
# ---------------------------------------------------------------------------


def test_run_hyperswap_binds_native_mask_output():
    proc = _RecordingProc()
    fs = _make_swappers(proc, _make_model(["output", "mask"]))

    mask = torch.empty(1, 1, 256, 256)
    filled = fs.run_hyperswap(
        torch.zeros(1, 3, 256, 256),
        torch.zeros(1, 512),
        torch.empty(1, 3, 256, 256),
        "HyperSwap-v3",
        mask_output=mask,
    )

    assert filled is True
    assert proc.bound_outputs == ["output", "mask"]
    assert fs.runs == 1
    assert fs.hyperswap_native_mask_ready() is True


def test_run_hyperswap_without_mask_request_binds_only_output():
    proc = _RecordingProc()
    fs = _make_swappers(proc, _make_model(["output", "mask"]))

    filled = fs.run_hyperswap(
        torch.zeros(1, 3, 256, 256),
        torch.zeros(1, 512),
        torch.empty(1, 3, 256, 256),
        "HyperSwap-v3",
    )

    assert filled is False
    assert proc.bound_outputs == ["output"]
    # Not requesting the mask must not mark it as unavailable.
    assert fs.hyperswap_native_mask_ready() is True


def test_native_mask_disabled_when_export_has_single_output():
    proc = _RecordingProc()
    fs = _make_swappers(proc, _make_model(["output"]))

    filled = fs.run_hyperswap(
        torch.zeros(1, 3, 256, 256),
        torch.zeros(1, 512),
        torch.empty(1, 3, 256, 256),
        "HyperSwap-v3",
        mask_output=torch.empty(1, 1, 256, 256),
    )

    assert filled is False
    assert fs.hyperswap_native_mask_ready() is False
    # The swap itself still ran, exactly once.
    assert proc.bound_outputs == ["output"]
    assert fs.runs == 1


def test_native_mask_falls_back_to_maskless_run_when_binding_fails():
    proc = _RecordingProc(fail_on_output="mask")
    fs = _make_swappers(proc, _make_model(["output", "mask"]))

    filled = fs.run_hyperswap(
        torch.zeros(1, 3, 256, 256),
        torch.zeros(1, 512),
        torch.empty(1, 3, 256, 256),
        "HyperSwap-v3",
        mask_output=torch.empty(1, 1, 256, 256),
    )

    assert filled is False
    assert fs.hyperswap_native_mask_ready() is False
    # First attempt bound "output" then failed on "mask"; retry bound "output" again.
    assert proc.bound_outputs == ["output", "output"]
    assert fs.runs == 1


def test_maskless_run_still_propagates_errors():
    """Without a mask request the original raising behaviour is preserved."""
    proc = _RecordingProc(fail_on_output="output")
    fs = _make_swappers(proc, _make_model(["output", "mask"]))

    with pytest.raises(RuntimeError):
        fs.run_hyperswap(
            torch.zeros(1, 3, 256, 256),
            torch.zeros(1, 512),
            torch.empty(1, 3, 256, 256),
            "HyperSwap-v3",
        )


def test_run_hyperswap_batched_binds_native_mask():
    proc = _RecordingProc()
    fs = _make_swappers(proc, _make_model(["output", "mask"]))

    images = torch.zeros(3, 3, 256, 256)
    ok = fs.run_hyperswap_batched(
        images,
        torch.zeros(1, 512),
        torch.empty_like(images),
        "HyperSwap-v3",
        mask_output=torch.empty(3, 1, 256, 256),
    )

    assert ok is True
    assert proc.bound_outputs == ["output", "mask"]
    assert fs.hyperswap_native_mask_ready() is True


def test_run_hyperswap_batched_survives_mask_rejection():
    """A mask the engine rejects must not disable the whole batched path."""
    proc = _RecordingProc(fail_on_output="mask")
    fs = _make_swappers(proc, _make_model(["output", "mask"]))

    images = torch.zeros(2, 3, 256, 256)
    ok = fs.run_hyperswap_batched(
        images,
        torch.zeros(2, 512),
        torch.empty_like(images),
        "HyperSwap-v3",
        mask_output=torch.empty(2, 1, 256, 256),
    )

    assert ok is True
    assert fs.hyperswap_native_mask_ready() is False
    assert fs._hyperswap_ort_batch_session_disabled is False


# ---------------------------------------------------------------------------
# frame_worker: gating and mask composition
# ---------------------------------------------------------------------------


def _worker_stub(ready: bool = True):
    from app.processors.workers.frame_worker import FrameWorker

    def _resize(h, w):
        def _apply(t):
            return torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            )[0]

        return _apply

    return types.SimpleNamespace(
        HYPERSWAP_MODELS=FrameWorker.HYPERSWAP_MODELS,
        models_processor=types.SimpleNamespace(
            hyperswap_native_mask_ready=lambda: ready
        ),
        _get_cached_resize_bilinear_aa=_resize,
    )


def _strength(params, swapper_model="HyperSwap-v3", ready=True):
    from app.processors.workers.frame_worker import FrameWorker

    return FrameWorker._hyperswap_native_mask_strength(
        _worker_stub(ready), swapper_model, params
    )


def test_native_mask_strength_off_for_other_swappers():
    params = {
        "HyperSwapNativeMaskEnableToggle": True,
        "HyperSwapNativeMaskStrengthSlider": 100,
    }
    assert _strength(params, swapper_model="Inswapper128") == 0.0
    assert _strength(params, swapper_model="GhostFace-v2") == 0.0


def test_native_mask_strength_off_when_toggle_off_or_missing():
    assert _strength({}) == 0.0
    assert _strength({"HyperSwapNativeMaskEnableToggle": False}) == 0.0


def test_native_mask_strength_off_when_model_reports_unavailable():
    params = {
        "HyperSwapNativeMaskEnableToggle": True,
        "HyperSwapNativeMaskStrengthSlider": 100,
    }
    assert _strength(params, ready=False) == 0.0


def test_native_mask_strength_scaled_and_clamped():
    def p(value):
        return {
            "HyperSwapNativeMaskEnableToggle": True,
            "HyperSwapNativeMaskStrengthSlider": value,
        }

    assert _strength(p(100)) == pytest.approx(1.0)
    assert _strength(p(50)) == pytest.approx(0.5)
    assert _strength(p(0)) == pytest.approx(0.0)
    assert _strength(p(500)) == pytest.approx(1.0)
    assert _strength(p(-10)) == pytest.approx(0.0)
    assert _strength(p("bogus")) == 0.0
    # Default is full strength once the toggle is on.
    assert _strength({"HyperSwapNativeMaskEnableToggle": True}) == pytest.approx(1.0)


def _apply(swap_mask, native_mask, strength, ready=True):
    from app.processors.workers.frame_worker import FrameWorker

    return FrameWorker._apply_hyperswap_native_mask(
        _worker_stub(ready), swap_mask, native_mask, strength
    )


def test_apply_native_mask_renormalizes_peak_to_one():
    """The raw output rarely peaks at exactly 1.0; the interior must stay fully opaque."""
    swap_mask = torch.ones(1, 8, 8)
    native = torch.full((1, 1, 8, 8), 0.9)
    native[0, 0, 0, 0] = 0.0

    out = _apply(swap_mask, native, 1.0)

    assert out.shape == swap_mask.shape
    assert out[0, 1, 1] == pytest.approx(1.0)
    assert out[0, 0, 0] == pytest.approx(0.0)


def test_apply_native_mask_zeroes_excluded_region():
    swap_mask = torch.ones(1, 4, 4)
    native = torch.ones(1, 1, 4, 4)
    native[0, 0, :, 2:] = 0.0

    out = _apply(swap_mask, native, 1.0)

    assert torch.allclose(out[0, :, :2], torch.ones(4, 2))
    assert torch.allclose(out[0, :, 2:], torch.zeros(4, 2))


def test_apply_native_mask_partial_strength_interpolates():
    swap_mask = torch.ones(1, 4, 4)
    native = torch.ones(1, 1, 4, 4)
    native[0, 0, :, 2:] = 0.0

    out = _apply(swap_mask, native, 0.25)

    # strength 0.25 keeps 75% of the swap where the native mask is 0.
    assert torch.allclose(out[0, :, 2:], torch.full((4, 2), 0.75))
    assert torch.allclose(out[0, :, :2], torch.ones(4, 2))


def test_apply_native_mask_ignores_degenerate_prediction():
    swap_mask = torch.ones(1, 4, 4) * 0.5
    native = torch.full((1, 1, 4, 4), 0.01)

    out = _apply(swap_mask, native, 1.0)

    assert torch.equal(out, swap_mask)


def test_apply_native_mask_resizes_to_swap_mask():
    swap_mask = torch.ones(1, 16, 16)
    native = torch.ones(1, 1, 8, 8)

    out = _apply(swap_mask, native, 1.0)

    assert out.shape == (1, 16, 16)
    assert torch.allclose(out, torch.ones(1, 16, 16))


def test_apply_native_mask_does_not_mutate_inputs():
    swap_mask = torch.ones(1, 4, 4)
    native = torch.full((1, 1, 4, 4), 0.8)
    native_before = native.clone()

    out = _apply(swap_mask, native, 1.0)

    assert torch.equal(native, native_before)
    assert torch.equal(swap_mask, torch.ones(1, 4, 4))
    assert out is not swap_mask


# ---------------------------------------------------------------------------
# UI wiring
# ---------------------------------------------------------------------------


def test_native_mask_controls_declared_for_all_hyperswap_versions():
    # Avoid importing SWAPPER_LAYOUT_DATA (circular UI imports under pytest).
    from pathlib import Path

    text = Path("app/ui/widgets/swapper_layout_data.py").read_text(encoding="utf-8")
    assert '"HyperSwapNativeMaskEnableToggle"' in text
    assert '"HyperSwapNativeMaskStrengthSlider"' in text
    block = text.split('"HyperSwapNativeMaskEnableToggle"', 1)[1].split(
        '"InStyleResAEnableToggle"', 1
    )[0]
    for name in ("HyperSwap-v1", "HyperSwap-v2", "HyperSwap-v3"):
        assert block.count(f'"{name}"') == 2, name


def test_selection_value_matches_supports_single_value_and_lists():
    from app.ui.widgets.actions.common_actions import selection_value_matches

    assert selection_value_matches({"requiredSelectionValue": "Inswapper128"}, "Inswapper128")
    assert not selection_value_matches(
        {"requiredSelectionValue": "Inswapper128"}, "HyperSwap-v1"
    )

    multi = {"requiredSelectionValue": ["HyperSwap-v1", "HyperSwap-v3"]}
    assert selection_value_matches(multi, "HyperSwap-v1")
    assert selection_value_matches(multi, "HyperSwap-v3")
    assert not selection_value_matches(multi, "HyperSwap-v2")
    assert not selection_value_matches(multi, "Inswapper128")

    # A row without the key stays hidden rather than matching everything.
    assert not selection_value_matches({}, "Inswapper128")

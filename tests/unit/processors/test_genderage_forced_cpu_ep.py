"""GenderAge must stay on the CPU EP.

On the CUDA EP every depthwise Conv of ``genderage.onnx`` falls back to a slow
cuDNN path ("running in Fallback mode. May be extremely slow."), measured at
~233 ms per call versus ~0.6 ms on the CPU EP for the same batch of 3 faces.
Routing this model back to CUDA silently destroys playback performance, so pin
the behaviour down here.
"""

from __future__ import annotations

import types

import pytest

mp = pytest.importorskip(
    "app.processors.models_processor",
    reason="models_processor pulls the full ML stack (einops, kornia, ...)",
)


def _providers_for(model_name: str, providers, device: str = "cuda"):
    stub = types.SimpleNamespace(providers=providers, device=device)
    return mp.ModelsProcessor._providers_for_onnx_model(stub, model_name)


def test_genderage_is_in_force_cpu_list():
    assert "GenderAge" in mp.ONNX_MODELS_FORCE_CPU_EP


def test_genderage_gets_cpu_ep_under_tensorrt_provider():
    providers = _providers_for(
        "GenderAge", [("TensorrtExecutionProvider", {}), "CUDAExecutionProvider"]
    )
    assert providers == [("CPUExecutionProvider")]


def test_genderage_gets_cpu_ep_under_plain_cuda_provider():
    providers = _providers_for("GenderAge", ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert providers == [("CPUExecutionProvider")]


def test_other_models_are_unaffected():
    app_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert _providers_for("Inswapper128", app_providers) is app_providers


def test_cpu_ep_thread_budget_is_bounded():
    # Must not grab every core away from the video pipeline.
    assert 1 <= mp.CPU_EP_MODEL_INTRA_OP_THREADS <= 4

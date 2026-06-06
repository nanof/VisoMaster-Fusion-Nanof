"""ORT IOBinding device_type normalization."""

from app.processors.ort_io_dtype_utils import normalize_ort_bind_device_type


def test_normalize_ort_bind_device_type_cuda_index():
    assert normalize_ort_bind_device_type("cuda:0") == "cuda"
    assert normalize_ort_bind_device_type("cuda:1") == "cuda"
    assert normalize_ort_bind_device_type("CUDA:2") == "cuda"


def test_normalize_ort_bind_device_type_cpu():
    assert normalize_ort_bind_device_type("cpu") == "cpu"

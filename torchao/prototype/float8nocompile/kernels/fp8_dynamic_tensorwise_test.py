import pytest
import torch
from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
from torchao.float8.float8_tensor import LinearMMConfig
from torchao.prototype.float8nocompile.kernels.fp8_dynamic_tensorwise import (
    KernelAlgorithm,
    triton_hp_tensor_to_float8_dynamic,
)


@pytest.mark.parametrize(
    "algo", [KernelAlgorithm.ATOMIC_MAX, KernelAlgorithm.REDUCTION]
)
@pytest.mark.parametrize(
    "input_shape",
    [(32, 32), (512, 512), (4096, 4096)],
)
def test_fp8_triton_hp_tensor_to_float8_dynamic(
    algo: KernelAlgorithm, input_shape: tuple[int, int]
):
    assert torch.cuda.is_available()
    device = "cuda"
    input_bf16 = torch.randn(input_shape, dtype=torch.bfloat16, device=device)
    x_bf16 = input_bf16.clone().detach().to(device)
    y_bf16 = input_bf16.clone().detach().to(device)

    # production implementation
    x_fp8 = hp_tensor_to_float8_dynamic(
        x_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
    )

    # float8nocompile triton implementation
    y_fp8 = triton_hp_tensor_to_float8_dynamic(
        y_bf16,
        torch.float8_e4m3fn,
        LinearMMConfig(),
        algo=algo,
    )

    def allclose_fp8(tensor1, tensor2, atol=1e-3, rtol=1e-3):
        # convert fp8 tensors to a higher precision (e.g., float32) for comparison
        # since torch.allclose does not support fp8 tensors
        if tensor1.shape != tensor2.shape:
            raise ValueError("Tensors must have the same shape for comparison.")
        if tensor1.dtype != tensor2.dtype:
            raise ValueError("Tensors must have the same dtype for comparison.")

        tensor1_fp32 = tensor1.to(torch.float32)
        tensor2_fp32 = tensor2.to(torch.float32)
        return torch.allclose(tensor1_fp32, tensor2_fp32, atol=atol, rtol=rtol)

    assert torch.allclose(x_fp8._scale, y_fp8._scale, atol=1e-3, rtol=1e-3)
    assert allclose_fp8(x_fp8._data, y_fp8._data, atol=1e-3, rtol=1e-3)

    # assert that error is raised when input tensor is not contiguous
    with pytest.raises(AssertionError, match="tensor must be contiguous"):
        triton_hp_tensor_to_float8_dynamic(
            y_bf16.t(),  # transpose so tensor memory layout is no longer contiguous
            torch.float8_e4m3fn,
            LinearMMConfig(),
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton kernels for scaling high precision tensors to float8.
"""
from enum import Enum

import torch
import triton
import triton.language as tl

from torchao.float8.float8_tensor import Float8Tensor, GemmInputRole, LinearMMConfig

EPS = 1e-12

FP8_DTYPE_MAP = {
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


class KernelAlgorithm(Enum):
    """Enum for FP8 conversion strategy."""

    # use atomic max to compute global amax between blocks
    ATOMIC_MAX = "atomic_max"

    # reduce shared buffer containing local block amaxes to find global amax
    REDUCTION = "reduction"


class MemoryLayout(Enum):
    """Enum for memory layout of input tensor."""

    # return 1 output tensor with row-major memory layout
    ROW_MAJOR = "row_major"

    # return 1 output tensor with column-major memory layout
    COL_MAJOR = "col_major"

    # return 2 output tensors, one with row-major and the other with column-major memory layout
    ROW_AND_COL_MAJOR = "ROW_AND_COL_MAJOR"


kernel_configs = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
]


# --- atomic max version of kernel ---
@triton.autotune(configs=kernel_configs, key=["num_elements"])
@triton.jit
def _block_amax_atomic(
    input_ptr,
    amax_ptr,
    num_elements,
    input_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    # compute local amax for each block
    block_id = tl.program_id(axis=0)
    block_start = block_id * BLOCK_SIZE
    block_offs = block_start + tl.arange(0, BLOCK_SIZE)
    block_mask = block_offs < num_elements
    vals = tl.load(input_ptr + block_offs, mask=block_mask).to(input_dtype)
    block_amax = tl.max(tl.abs(vals))
    tl.atomic_max(amax_ptr, block_amax)


@triton.jit
def _fp8_scale_atomic(
    amax_ptr,
    scale_out_ptr,
    fp8_dtype_max,
    EPS: tl.constexpr,
):
    # load previously computed global amax
    global_amax = tl.load(amax_ptr)

    # compute scale, must be fp32
    scale = (fp8_dtype_max / tl.clamp(global_amax, min=EPS, max=float("inf"))).to(
        tl.float32
    )

    # store scale for use in Float8Tensor constructor
    scale_off = tl.arange(0, 1)
    tl.store(scale_out_ptr + scale_off, scale)


@triton.autotune(configs=kernel_configs, key=["num_elements"])
@triton.jit
def _to_fp8_row_major(
    input_ptr,
    scale_ptr,
    out_ptr,
    num_elements,
    fp8_dtype_min,
    fp8_dtype_max,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    block_id = tl.program_id(axis=0)

    # load scale
    scale = tl.load(scale_ptr)

    # load block of input tensor
    block_start = block_id * BLOCK_SIZE
    block_offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = block_offs < num_elements
    vals = tl.load(input_ptr + block_offs, mask=mask).to(input_dtype)

    # perform conversion
    vals = vals * scale
    fp8_vals = tl.clamp(vals, min=fp8_dtype_min, max=fp8_dtype_max).to(output_dtype)
    tl.store(out_ptr + block_offs, fp8_vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_ROWS": 128, "BLOCK_SIZE_COLS": 128}, num_warps=1),
        triton.Config({"BLOCK_SIZE_ROWS": 256, "BLOCK_SIZE_COLS": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE_ROWS": 512, "BLOCK_SIZE_COLS": 512}, num_warps=4),
    ],
    key=["num_elements"],
)
@triton.jit
def _to_fp8_col_major(
    input_ptr,
    scale_ptr,
    out_ptr,
    num_elements,
    fp8_dtype_min,
    fp8_dtype_max,
    num_rows,
    num_cols,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
    EPS: tl.constexpr,
):
    block_row_id = tl.program_id(axis=0)
    block_col_id = tl.program_id(axis=1)

    # load scale
    scale = tl.load(scale_ptr)

    # load block of input tensor
    block_row_start = block_row_id * BLOCK_SIZE_ROWS
    block_col_start = block_col_id * BLOCK_SIZE_COLS
    block_row_offs = block_row_start + tl.arange(0, BLOCK_SIZE_ROWS)
    block_col_offs = block_col_start + tl.arange(0, BLOCK_SIZE_COLS)
    block_offs = block_row_offs[:, None] * num_cols + block_col_offs[None, :]
    mask = (block_row_offs[:, None] < num_rows) & (block_col_offs[None, :] < num_cols)
    vals = tl.load(input_ptr + block_offs, mask=mask).to(input_dtype)

    # perform conversion
    vals = vals * scale
    fp8_vals = tl.clamp(vals, min=fp8_dtype_min, max=fp8_dtype_max).to(output_dtype)
    out_offs = block_col_offs[None, :] * num_rows + block_row_offs[:, None]
    tl.store(out_ptr + out_offs, fp8_vals, mask=mask)


# --- reduction version of amax and scale computations ---
@triton.jit
def _block_amax_reduction(
    input_ptr,
    block_amaxes_ptr,
    num_elements,
    input_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    # compute local amax for each block
    block_id = tl.program_id(axis=0)
    block_start = block_id * BLOCK_SIZE
    block_offs = block_start + tl.arange(0, BLOCK_SIZE)
    block_mask = block_offs < num_elements
    vals = tl.load(input_ptr + block_offs, mask=block_mask).to(input_dtype)
    block_amax = tl.max(tl.abs(vals))
    tl.store(block_amaxes_ptr + block_id, block_amax)


@triton.jit
def _fp8_scale_reduction(
    block_amaxes_ptr,
    scale_out_ptr,
    num_elements,
    fp8_dtype_max,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    # calculate global amax across all blocks
    global_amax = tl.zeros([1], dtype=tl.float64)
    num_blocks = tl.cdiv(num_elements, BLOCK_SIZE)
    for i in range(num_blocks):
        block_max = tl.load(block_amaxes_ptr + i)
        global_amax = tl.maximum(global_amax, block_max)

    # compute scale, must be fp32
    scale = (fp8_dtype_max / tl.clamp(global_amax, min=EPS, max=float("inf"))).to(
        tl.float32
    )
    scale_off = tl.arange(0, 1)
    tl.store(scale_out_ptr + scale_off, scale)


def triton_hp_tensor_to_float8_dynamic(
    hp_tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
    algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
    memory_layout: MemoryLayout = MemoryLayout.ROW_MAJOR,
) -> Float8Tensor:

    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[fp8_dtype]

    fp8_dtype_min = torch.finfo(fp8_dtype).min
    fp8_dtype_max = torch.finfo(fp8_dtype).max

    # compute scaling factor for tensor
    scale = _triton_hp_tensor_to_scale(
        hp_tensor,
        tl_input_dtype,
        fp8_dtype_max,
        algo,
    )

    # perform fp8 conversion
    fp8_tensor_row_major, fp8_tensor_col_major = _triton_hp_tensor_and_scale_to_fp8(
        hp_tensor,
        scale,
        fp8_dtype,
        fp8_dtype_min,
        fp8_dtype_max,
        tl_input_dtype,
        tl_output_dtype,
        linear_mm_config,
        gemm_input_role,
        memory_layout,
    )
    return fp8_tensor_row_major, fp8_tensor_col_major


def _triton_hp_tensor_to_scale(
    hp_tensor: torch.Tensor,
    tl_input_dtype: tl.core.dtype,
    fp8_dtype_max: float,
    algo: KernelAlgorithm,
):
    num_elements = hp_tensor.numel()
    scale_out = torch.empty((1,), dtype=torch.float32, device=hp_tensor.device)
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    # compute the fp8 scale using the given algorithm
    if algo == KernelAlgorithm.ATOMIC_MAX:
        global_amax = torch.zeros((1,), dtype=torch.float32, device=hp_tensor.device)
        # compute global amax to be used for scaling
        _block_amax_atomic[grid](
            hp_tensor,
            global_amax,
            num_elements,
            input_dtype=tl_input_dtype,
            EPS=EPS,
        )

        # compute scale for fp8 conversion
        _fp8_scale_atomic[1, 1, 1](
            global_amax,
            scale_out,
            fp8_dtype_max,
            EPS=EPS,
        )

    elif algo == KernelAlgorithm.REDUCTION:
        # max block size and num warps values determined via manual tuning
        max_block_size = 4096
        num_warps = 8
        block_size = min(max_block_size, num_elements)
        block_amaxes = torch.zeros(
            (num_elements // block_size,), dtype=torch.float32, device=hp_tensor.device
        )
        # compute local amax for each block
        _block_amax_reduction[grid](
            hp_tensor,
            block_amaxes,
            num_elements,
            input_dtype=tl_input_dtype,
            BLOCK_SIZE=block_size,
            EPS=EPS,
            num_warps=num_warps,
        )

        # calculate global amax across all blocks and use it to compute scale
        _fp8_scale_reduction[(1, 1, 1)](
            block_amaxes,
            scale_out,
            num_elements,
            fp8_dtype_max,
            BLOCK_SIZE=block_size,
            EPS=EPS,
        )
    else:
        raise ValueError(f"Unsupported kernel algorithm: {algo}")

    return scale_out


def _triton_hp_tensor_and_scale_to_fp8(
    hp_tensor: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
    fp8_dtype_min: float,
    fp8_dtype_max: float,
    tl_input_dtype: tl.core.dtype,
    tl_output_dtype: tl.core.dtype,
    linear_mm_config: LinearMMConfig,
    gemm_input_role: GemmInputRole,
    memory_layout: MemoryLayout,
):
    orig_shape = hp_tensor.shape
    num_elements = hp_tensor.numel()
    fp8_tensor_row_major = None
    fp8_tensor_col_major = None

    # perform fp8 conversion with output in row major memory layout
    if memory_layout in {MemoryLayout.ROW_MAJOR, MemoryLayout.ROW_AND_COL_MAJOR}:
        fp8_output_row_major = torch.empty(
            orig_shape, dtype=fp8_dtype, device=hp_tensor.device
        )
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
        # launch triton kernel to perform conversion
        _to_fp8_row_major[grid](
            hp_tensor,
            scale,
            fp8_output_row_major,
            num_elements,
            fp8_dtype_min,
            fp8_dtype_max,
            input_dtype=tl_input_dtype,
            output_dtype=tl_output_dtype,
            EPS=EPS,
        )
        # wrap in Float8Tensor
        fp8_tensor_row_major = Float8Tensor(
            fp8_output_row_major,
            scale,
            orig_dtype=hp_tensor.dtype,
            linear_mm_config=linear_mm_config,
            gemm_input_role=gemm_input_role,
        )

    # perform fp8 conversion with output in column major memory layout
    if memory_layout in {MemoryLayout.COL_MAJOR, MemoryLayout.ROW_AND_COL_MAJOR}:
        fp8_output_col_major = torch.empty(
            orig_shape, dtype=fp8_dtype, device=hp_tensor.device
        )
        num_rows, num_cols = orig_shape
        grid = lambda meta: (
            triton.cdiv(num_rows, meta["BLOCK_SIZE_ROWS"]),
            triton.cdiv(num_cols, meta["BLOCK_SIZE_COLS"]),
        )
        # launch triton kernel to perform conversion
        _to_fp8_col_major[grid](
            hp_tensor,
            scale,
            fp8_output_col_major,
            num_elements,
            fp8_dtype_min,
            fp8_dtype_max,
            num_rows,
            num_cols,
            input_dtype=tl_input_dtype,
            output_dtype=tl_output_dtype,
            EPS=EPS,
        )

        # for col major we need to update the strides to reflect the new memory layout
        col_major_strides = (1, num_rows)
        fp8_output_col_major = fp8_output_col_major.as_strided(
            fp8_output_col_major.size(), col_major_strides
        )

        # wrap in Float8Tensor
        fp8_tensor_col_major = Float8Tensor(
            fp8_output_col_major,
            scale,
            orig_dtype=hp_tensor.dtype,
            linear_mm_config=linear_mm_config,
            gemm_input_role=gemm_input_role,
        )

    return fp8_tensor_row_major, fp8_tensor_col_major

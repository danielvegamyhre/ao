# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A simple module swap UX for a float8 version of `torch.nn.Linear` which
does not require `torch.compile` to be performant.
"""
import pdb
from typing import Optional

import torch

from torchao.float8.config import Float8LinearConfig, ScalingGranularity, ScalingType
from torchao.float8.distributed_utils import tensor_already_casted_to_fp8
from torchao.float8.float8_linear import manual_float8_matmul_with_args_in_float8
from torchao.float8.float8_scaling_utils import NoopFwToFloat8BwDynamic
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig, ScaledMMConfig
from torchao.float8.float8_utils import tensor_to_scale

from torchao.prototype.float8nocompile.float8nocompile_scaling_utils import (
    Float8Conversion,
    Float8ConversionColumnMajor,
    Float8ConversionRowMajor,
    NoopFwToFloat8NoCompileBwDynamic,
)
from torchao.prototype.float8nocompile.kernels.fp8_dynamic_tensorwise import (
    KernelAlgorithm,
    MemoryLayout,
)


class Float8LinearNoCompile(torch.nn.Linear):
    """
    Float8LinearNoCompile is a version of Float8Linear that does not require
    the use of torch.compile to be performant.

    Note: this is **prototype** and not suitable for production use.
    """

    def __init__(self, *args, **kwargs):
        """
        Additional arguments on top of `torch.nn.Linear`'s arguments:
        * `config`: Float8LinearConfig
        """
        config = kwargs.pop("config")
        kernel_algo = kwargs.pop("kernel_algo")
        emulate = config.emulate
        super().__init__(*args, **kwargs)

        self.config = config
        self.kernel_algo = kernel_algo

        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                emulate,
                self.config.gemm_config_output.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_input
            ScaledMMConfig(
                emulate,
                self.config.gemm_config_grad_input.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_weight
            ScaledMMConfig(
                emulate,
                self.config.gemm_config_grad_weight.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # TODO(danielvegamyhre): support for FSDP once dependencies are implemented
        input_fp8_row_major = Float8ConversionRowMajor.apply(
            input,
            self.config.cast_config_input.target_dtype,
            self.linear_mm_config,
            GemmInputRole.INPUT,
            self.kernel_algo,
        )

        weight_t_fp8_col_major = Float8ConversionColumnMajor.apply(
            self.weight.t().contiguous(),  # contiguous inputs required for triton kernels
            self.config.cast_config_weight.target_dtype,
            self.linear_mm_config,
            GemmInputRole.WEIGHT,
            self.kernel_algo,
        )

        # output = matmul_with_args_in_hp.apply(
        #     input,
        #     self.weight.t(),
        #     self.config,
        #     self.linear_mm_config,
        #     self.kernel_algo,
        # )
        output = matmul_with_args_in_fp8.apply(
            input_fp8_row_major,
            weight_t_fp8_col_major,
        )

        # cast grad_output to float8_e5m2 during backward
        output = NoopFwToFloat8NoCompileBwDynamic.apply(
            output,
            self.config.cast_config_grad_output.target_dtype,
            self.linear_mm_config,
            self.kernel_algo,
        )
        return output

    @classmethod
    def from_float(cls, mod, kernel_algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            config (Optional[Float8LinearConfig]): configuration for conversion to float8
        """
        config = Float8LinearConfig()
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
                config=config,
                kernel_algo=kernel_algo,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias

        # TODO(danielvegamyhre): support for FSDP once dependencies are implemented
        return new_mod


class matmul_with_args_in_fp8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_fp8_row_major,
        weight_t_fp8_col_major,
    ):
        ctx.save_for_backward(input_fp8_row_major, weight_t_fp8_col_major)
        output = torch.mm(input_fp8_row_major, weight_t_fp8_col_major)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_fp8_row_major, weight_t_fp8_col_major = ctx.saved_tensors

        # grad_input = grad_output @ weight (backward pass)
        grad_input = torch.mm(
            grad_output,
            weight_t_fp8_col_major.t(),
        )

        # grad_weight = input_t @ grad_output (backward pass)
        grad_weight = torch.mm(grad_output.t(), input_fp8_row_major)
        return grad_input, grad_weight.t()


class matmul_with_args_in_hp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_hp, weight_t_hp, config, linear_mm_config, kernel_algo):
        ctx.save_for_backward(input_hp, weight_t_hp)
        ctx.config = config
        ctx.linear_mm_config = linear_mm_config
        ctx.kernel_algo = kernel_algo

        # input @ weight_t = output (forward pass)
        input_fp8_row_major = Float8ConversionRowMajor.apply(
            input_hp,
            config.cast_config_input.target_dtype,
            linear_mm_config,
            GemmInputRole.INPUT,
            kernel_algo,
        )

        weight_t_fp8_col_major = Float8ConversionColumnMajor.apply(
            weight_t_hp.contiguous(),  # triton kernel requires input tensor be contiguous
            config.cast_config_weight.target_dtype,
            linear_mm_config,
            GemmInputRole.WEIGHT,
            kernel_algo,
        )

        output = torch.mm(input_fp8_row_major, weight_t_fp8_col_major)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad output is already e5m2
        input_hp, weight_t_hp = ctx.saved_tensors

        # 1. grad_input = grad_output @ weight (backward pass)
        weight_fp8_col_major = Float8ConversionColumnMajor.apply(
            weight_t_hp.t().contiguous(),  # triton kernel requires input tensor be contiguous
            ctx.config.cast_config_weight.target_dtype,
            ctx.linear_mm_config,
            GemmInputRole.WEIGHT,
            ctx.kernel_algo,
        )
        grad_input = torch.mm(grad_output, weight_fp8_col_major)

        # 2. grad_weight = input_t @ grad_output (backward pass)
        # TODO: find out why float8_linear.py uses grad_weight = grad_output_t @ input
        input_fp8_col_major = Float8ConversionColumnMajor.apply(
            input_hp,
            ctx.config.cast_config_input.target_dtype,
            ctx.linear_mm_config,
            GemmInputRole.INPUT,
            ctx.kernel_algo,
        )

        grad_weight = torch.mm(grad_output.t(), input_fp8_col_major)
        return grad_input, grad_weight.t(), None, None, None

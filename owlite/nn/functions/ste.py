""" Quantization functions has various gradients"""
from typing import Any

import torch
from torch.autograd import Function

from .fake_quantize import fake_quantize


# pylint: disable= abstract-method, arguments-differ
class RoundSTEFunction(Function):
    """Round function using STE."""

    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs)

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        return grad_outputs.clone()


class STEFunction(Function):
    """fake quantizing function for QAT using STE (Straight-Through Estimator)

    For quant_min <= input <= quant_max the gradient passes straight through,
    otherwise the gradient is zero
    """

    # pylint: disable=unused-argument, duplicate-code
    @staticmethod
    def forward(
        ctx: Any,
        inputs,
        step_size,
        zero_point,
        grad_scale,
        quant_min,
        quant_max,
        per_channel,
        compensate_zp,  # compensate_zp is unused argument in symmetric quantization
    ):
        """grad_scale and compensate_zp are unused arguments in symmetric quantization"""
        ctx.save_for_backward(inputs)
        ctx.other = quant_min, quant_max
        return fake_quantize(inputs, step_size, zero_point, quant_min, quant_max, per_channel)

    # pylint: enable= unused-argument

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        inputs = ctx.saved_tensors[0]
        quant_min, quant_max = ctx.other
        zero = grad_output.new_zeros(1)
        grad_inputs = torch.where((inputs >= quant_min) & (inputs <= quant_max), grad_output, zero)
        return grad_inputs, None, None, None, None, None, None, None


# pylint: enable= abstract-method, arguments-differ


round_ste = RoundSTEFunction.apply
ste_function = STEFunction.apply

from typing import Any, Union

import torch
from torch import BoolTensor, Tensor
from torch.autograd import Function

from .fake_quantize import fake_quantize


# pylint: disable= abstract-method, arguments-differ
class STEFunction(Function):
    """fake quantizing function for QAT using STE (Straight-Through Estimator)

    For quant_min <= input <= quant_max the gradient passes straight through,
    otherwise the gradient is zero
    """

    # pylint: disable=unused-argument, duplicate-code
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        step_size: Tensor,
        zero_point: Tensor,
        grad_scale: Tensor,
        quant_min: int,
        quant_max: int,
        per_channel: Union[bool, BoolTensor],
        compensate_zp: bool,  # compensate_zp is unused argument in symmetric quantization
    ) -> Any:
        """grad_scale and compensate_zp are unused arguments in symmetric quantization"""
        ctx.save_for_backward(inputs)
        lower_bound = quant_min * step_size
        upper_bound = quant_max * step_size
        ctx.other = lower_bound, upper_bound
        return fake_quantize(inputs, step_size, zero_point, quant_min, quant_max, per_channel)

    # pylint: enable= unused-argument

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        inputs = ctx.saved_tensors[0]
        grad_output = grad_outputs[0]
        lower_bound, upper_bound = ctx.other
        lower_bound = lower_bound.reshape([-1] + ([1] * (inputs.dim() - 1)))
        upper_bound = lower_bound.reshape([-1] + ([1] * (inputs.dim() - 1)))
        grad_inputs = torch.where(inputs.ge(lower_bound) * inputs.le(upper_bound), grad_output, 0)
        return grad_inputs, None, None, None, None, None, None, None


# pylint: enable= abstract-method, arguments-differ

ste_function = STEFunction.apply

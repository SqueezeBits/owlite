from typing import Any, Union

import torch
from torch import BoolTensor, Tensor
from torch.autograd import Function

from .fake_quantize import fake_quantize


# pylint: disable= abstract-method, arguments-differ
class CLQFunction(Function):
    """An implementation of QAT function using CLQ (Constrained Learned Quantization)"""

    # pylint: disable=unused-argument
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
        ctx.save_for_backward(inputs, step_size)
        ctx.other = grad_scale, quant_min, quant_max, per_channel
        return fake_quantize(inputs, step_size.abs(), zero_point, quant_min, quant_max, per_channel)

    # pylint: enable= unused-argument

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        inputs, step_size = ctx.saved_tensors
        grad_output = grad_outputs[0]
        step_size_abs = step_size.abs().reshape([-1] + ([1] * (inputs.dim() - 1)))
        grad_scale, quant_min, quant_max, per_channel = ctx.other
        affine_input = (inputs / step_size_abs).clip(quant_min, quant_max)
        between = affine_input.gt(quant_min) & affine_input.lt(quant_max)
        grad_step_size = (
            torch.where(between, (torch.round(affine_input) - affine_input), affine_input) * grad_output * grad_scale
        )
        grad_step_size = (
            grad_step_size.sum(dim=tuple(range(1, inputs.dim())), keepdim=False)
            if per_channel
            else grad_step_size.sum().unsqueeze(dim=0)
        )
        grad_step_size = grad_step_size * torch.where(step_size == 0.0, 1.0, step_size.sign())
        grad_output = grad_output * between
        return grad_output, grad_step_size, None, None, None, None, None, None


# pylint: enable= abstract-method, arguments-differ

clq_function = CLQFunction.apply

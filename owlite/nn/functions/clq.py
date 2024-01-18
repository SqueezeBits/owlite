from typing import Any

import torch
from torch.autograd import Function

from .fake_quantize import fake_quantize


# pylint: disable= abstract-method, arguments-differ
class CLQFunction(Function):
    """An implementation of QAT function using CLQ (Constrained Learned Quantization)"""

    # pylint: disable=unused-argument
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
        ctx.save_for_backward(inputs, step_size)
        ctx.other = grad_scale, quant_min, quant_max, per_channel
        return fake_quantize(inputs, step_size.abs(), zero_point, quant_min, quant_max, per_channel)

    # pylint: enable= unused-argument

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        inputs, step_size = ctx.saved_tensors
        step_size_abs: torch.Tensor = step_size.abs()
        grad_scale, quant_min, quant_max, per_channel = ctx.other
        if per_channel:
            sizes = inputs.size()
            inputs = inputs.contiguous().view(inputs.size()[0], -1)
            inputs = torch.transpose(inputs, 0, 1)
            step_size_abs = torch.broadcast_to(step_size_abs, inputs.size())
            affine_input = inputs / step_size_abs
            affine_input = torch.transpose(affine_input, 0, 1)
            affine_input = affine_input.contiguous().view(sizes)
        else:
            affine_input = inputs / step_size_abs
        smaller = (affine_input < quant_min).float()
        bigger = (affine_input > quant_max).float()
        between = 1.0 - smaller - bigger
        if per_channel:
            grad_step_size = (
                (smaller * quant_min + bigger * quant_max + between * (torch.round(affine_input) - affine_input))
                * grad_output
                * grad_scale
            )
            grad_step_size = grad_step_size.contiguous().view(grad_step_size.size()[0], -1).sum(dim=1)
        else:
            grad_step_size = (
                (
                    (smaller * quant_min + bigger * quant_max + between * (torch.round(affine_input) - affine_input))
                    * grad_output
                    * grad_scale
                )
                .sum()
                .unsqueeze(dim=0)
            )
        grad_step_size = grad_step_size * (step_size.sign() + 0.5).sign()
        grad_output = between * grad_output
        return grad_output, grad_step_size, None, None, None, None, None, None


# pylint: enable= abstract-method, arguments-differ

clq_function = CLQFunction.apply

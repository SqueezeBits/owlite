from typing import Any, Optional

import torch
from torch import Tensor
from torch.autograd import Function

from .fake_quantize import fake_quantize


# mypy: disable-error-code=override
# pylint: disable-next=abstract-method
class CLQPlusFunction(Function):
    """An implementation of QAT function using CLQ+"""

    @staticmethod  # pylint: disable-next=arguments-differ
    def forward(
        ctx: Any,
        inputs: Tensor,
        step_size: Tensor,
        zero_point: Tensor,
        grad_scale: float,
        quant_min: int,
        quant_max: int,
        axis: Optional[int],
        compensate_zp: bool,
    ) -> Tensor:
        ctx.save_for_backward(inputs, step_size, zero_point)
        ctx.other = grad_scale, quant_min, quant_max, axis
        if axis:
            for _ in range(inputs.dim() - 1):
                zero_point = zero_point.unsqueeze(-1)
        inputs = inputs - zero_point

        fq_output = fake_quantize(inputs, step_size.abs(), torch.zeros_like(zero_point), quant_min, quant_max, axis)

        if compensate_zp:
            fq_output += zero_point
        return fq_output

    @staticmethod  # pylint: disable-next=arguments-differ
    def backward(ctx: Any, grad_output: Any) -> Any:
        inputs, step_size, zero_point = ctx.saved_tensors
        abs_step_size = step_size.abs()
        grad_scale, quant_min, quant_max, per_channel = ctx.other
        if per_channel:
            sizes = inputs.size()
            inputs = inputs.contiguous().view(inputs.size()[0], -1)
            inputs = torch.transpose(inputs, 0, 1)
            abs_step_size = torch.broadcast_to(abs_step_size, inputs.size())
            affine_input = (inputs - zero_point) / abs_step_size
            affine_input = torch.transpose(affine_input, 0, 1)
            affine_input = affine_input.contiguous().view(sizes)
        else:
            affine_input = (inputs - zero_point) / abs_step_size
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
            grad_zero_point = (smaller + bigger) * grad_output * grad_scale
            grad_zero_point = grad_zero_point.contiguous().view(grad_zero_point.size()[0], -1).sum(dim=1)
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
            grad_zero_point = ((smaller + bigger) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_output = between * grad_output
        grad_step_size = grad_step_size * (step_size.sign() + 0.5).sign()
        return grad_output, grad_step_size, grad_zero_point, None, None, None, None, None


clq_plus_function = CLQPlusFunction.apply

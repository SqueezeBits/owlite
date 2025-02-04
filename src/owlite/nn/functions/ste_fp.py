# pylint: disable=duplicate-code, unused-argument
from typing import Any

import torch
from torch import Tensor

from .fake_fp_quantize import BaseFakeFPQuantizeFunction, fake_fp8_quantize


# pylint: disable-next=abstract-method
class FakeFPQuantizeSTEFunction(BaseFakeFPQuantizeFunction):
    r"""Fake FP8 quantizing function for QAT using STE (Straight-Through Estimator).

    For $$ quant\_min $$ <= `input` <= $$ quant\_max $$ the gradient passes straight through,
    otherwise the gradient is zero

    When $$x$$ is input of FakeQuantize .

    $$
    \hat{x} = \text{FakeFPQuantize}(x)
    $$
    """

    @staticmethod  # pylint: disable-next=arguments-differ, too-many-positional-arguments
    def forward(
        ctx: Any,
        inputs: Tensor,
        step_size: Tensor,
        zero_point: Tensor,
        grad_scale: float,  # grad_scale is not used
        quant_min: float,
        quant_max: float,
        axis: int | None,
    ) -> Tensor:
        ctx.save_for_backward(inputs)
        lower_bound = quant_min * step_size
        upper_bound = quant_max * step_size
        ctx.other = lower_bound, upper_bound
        return fake_fp8_quantize(inputs, step_size, zero_point, quant_min=quant_min, quant_max=quant_max, axis=axis)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        inputs = ctx.saved_tensors[0]
        grad_output = grad_outputs[0]
        lower_bound, upper_bound = ctx.other
        lower_bound = lower_bound.reshape([-1] + ([1] * (inputs.dim() - 1)))
        upper_bound = lower_bound.reshape([-1] + ([1] * (inputs.dim() - 1)))
        grad_inputs = torch.where(inputs.ge(lower_bound) * inputs.le(upper_bound), grad_output, 0)
        return grad_inputs, None, None, None, None, None, None, None


fake_fp_quantize_ste_function = FakeFPQuantizeSTEFunction.apply

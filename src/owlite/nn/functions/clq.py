# pylint: disable=unused-argument
from typing import Any, Optional

import torch
from torch import Tensor
from torch.autograd import Function

from .fake_quantize import fake_quantize


# mypy: disable-error-code=override
# pylint: disable-next=abstract-method
class CLQFunction(Function):
    r"""An implementation of QAT function using CLQ (Constrained Learned Quantization)
    In **CLQ(Constrained Learned Quantization)** method, instead of using a fixed set of quantization levels,
    this method adapts the scales during training to minimize the impact on model performance. Learnable step_size
    allows the model to be better adapted to the distribution of fed data.
    ### Gradient of step\_size

    When $$x$$ is input of $$FakeQuantize$$ and $$s$$ is step\_size of $$FakeQuantize$$

    $$
    \dfrac{\partial \hat{x}}{\partial s}= \begin{cases} \left( -\dfrac{x}{|s|}+\left\lceil{\dfrac{x}{|s|}}
    \right\rfloor \right) \cdot \text{sign}(s) & \text{if, } \text{quant\_min} < \dfrac{x}{|s|} < \text{qant\_max}
    \\ \\ \text{quant\_min} \cdot \text{sign}(s) &\text{if, }\dfrac{x}{|s|}\leq \text{quant\_min} \\
    \\ \text{quant\_max}\cdot \text{sign}(s) &\text{if, } \dfrac{x}{|s|}\geq \text{quant\_max} \end{cases}
    $$
    """

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
        compensate_zp: bool,  # compensate_zp is not used
    ) -> Tensor:
        ctx.save_for_backward(inputs, step_size)
        ctx.other = grad_scale, quant_min, quant_max, axis
        return fake_quantize(inputs, step_size.abs(), zero_point, quant_min, quant_max, axis)

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


clq_function = CLQFunction.apply

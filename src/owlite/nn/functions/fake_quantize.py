# pylint: disable=unused-argument

from collections.abc import Callable

import torch
from torch import Tensor
from torch._C._onnx import TensorProtoDataType
from torch.autograd import Function
from torch.onnx._internal import jit_utils


def fake_quantize(
    inputs: Tensor,
    step_size: Tensor,
    zero_point: Tensor,
    *,
    quant_min: int,
    quant_max: int,
    axis: int | None = None,
) -> torch.Tensor:
    r"""Apply fake quantization function to the input with given quantization parameters.

    Equivalent to `torch.fake_quantize_per_channel_affine` if `per_channel` is `True`,
    `torch.fake_quantize_per_tensor_affine` otherwise.
    In OwLite, quantization is simulated through the following mathematical expression:

    $$$
    \small


        \text{FakeQuantize}(\text{input})=
        \left(
            \text{clip} \left(
                {\biggl\lfloor \frac{\text{input}}{\text{step\_size}} \biggr\rceil } + \text{zero\_point},
                \text{quant\_min},
                \text{quant\_max}
            \right) - \text{zero\_point}
        \right) \times \text{step\_size}
    $$$

    The primary objective of exporting to the Open Neural Network Exchange (ONNX) format is to facilitate deployment
    on TensorRT rather than the ONNX runtime. Consequently, the export process is confined to transforming the model
    into a format compatible with TensorRT, specifically one that supports fake quantization.
    The incorporation of fake quantization involves the decomposition of the model into `QuantizeLinear` and
    `DequantizeLinear` operations within the ONNX specification. Subsequently, TensorRT is entrusted with the task
    of ingesting the resultant ONNX graph and executing it in INT8 format, optimizing the process to the fullest extent
    of its capabilities. For more information, see the [TensorRT Developer Guide's section on Explicit
    Quantization](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#qat-models-work).

    Args:
        inputs (`torch.Tensor`): A tensor to quantize.
        step_size (`torch.Tensor`): The quantization scale, determining the magnitude of each quantization interval.
        zero_point (`torch.Tensor`): The quantization zero\\_point. It may be expressed as a float in the context of
                                        asymmetric quantization, while for symmetric quantization, it is fixed at 0.
        quant_min (`int`): The lower bound of the quantized domain, specified as an integer.
        quant_max (`int`): The upper bound of the quantized domain in as an integer.
        axis (`int`, optional): Channel axis. Only used when `per_channel` is `True`. Defaults to 0.

    Returns:
        torch.Tensor: fake-quantized tensor

    """
    # torch.fake_quantize_per_*_affine do not support bf16 as input
    if is_bfloat16 := inputs.dtype == torch.bfloat16:
        inputs = inputs.float()
    if axis is not None:
        output = torch.fake_quantize_per_channel_affine(
            inputs,
            step_size.float(),
            zero_point,
            axis,
            quant_min,
            quant_max,
        )
    else:
        output = torch.fake_quantize_per_tensor_affine(
            inputs,
            step_size.float(),
            # `torch.fake_quantize_per_tensor_affine` expects `zero_point` to be either int32 or int64
            # (See https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_tensor_affine.html)
            # while `torch.fake_quantize_per_channel_affine` doesn't
            zero_point,
            quant_min=quant_min,
            quant_max=quant_max,
        )
    if is_bfloat16:
        output = output.to(torch.bfloat16)
    return output


# pylint: disable-next=abstract-method
class BaseFakeINTQuantizeFunction(Function):
    """An autograd function for fake INT quantization.

    Static Methods:
        symbolic: Defines the symbolic computation graph for the function.
    """

    @staticmethod
    @torch.onnx.symbolic_helper.parse_args("v", "v", "v", "none", "i", "i", "i")  # type: ignore # pylint: disable-next=too-many-positional-arguments
    def symbolic(
        g: jit_utils.GraphContext,
        inputs: torch.Value,
        step_size: torch.Value,
        zero_point: torch.Value,
        grad_scale: float,
        quant_min: int,
        quant_max: int,
        axis: int | None,
    ) -> torch.Value | tuple[torch.Value, ...]:
        r"""Define the symbolic computation graph for the INT8 quantization.

        Args:
            g (`jit_utils.GraphContext`): The graph context.
            inputs (`torch.Value`): A tensor to quantize.
            step_size (`torch.Value`): The quantization scale, determining the magnitude of each quantization interval.
            zero_point (`torch.Value`): The quantization zero\_point. It may be expressed as a float in the context of
                asymmetric quantization, while for symmetric quantization, it is fixed at 0.
            grad_scale (`float`): The gradient scale.
            quant_min (`int`): The lower bound of the quantized domain, specified as an integer.
            quant_max (`int`): The upper bound of the quantized domain in as an integer.
            axis (`int`, optional): Channel axis. Only used when `per_channel` is `True`. Defaults to 0.

        Returns:
            The output value.
        """
        if (quant_min, quant_max) not in [(0, 255), (-128, 127)]:
            raise torch.onnx.errors.SymbolicValueError(
                "For int quantizer's (quant_min, quant_max), ONNX allows only (0, 255) and (-128, 127). "
                f"Got ({quant_min}, {quant_max})",
                inputs,
            )
        if quant_min == 0:
            zero_point = g.op("Cast", zero_point, to_i=TensorProtoDataType.UINT8)
        else:
            zero_point = g.op("Cast", zero_point, to_i=TensorProtoDataType.INT8)
        quantized = g.op("QuantizeLinear", inputs, step_size, zero_point, axis_i=axis)
        dequantized = g.op("DequantizeLinear", quantized, step_size, zero_point, axis_i=axis)
        return dequantized


FakeQuantizeSignature = Callable[
    [
        Tensor,  # inputs
        Tensor,  # step_size
        Tensor,  # zp
        float,  # grad_scale
        int | float,  # quant_min
        int | float,  # quant_max
        int | None,  # axis
    ],
    Tensor,
]

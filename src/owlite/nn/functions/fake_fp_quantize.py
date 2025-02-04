"""This module provides a custom PyTorch function for fake FP8 quantization.

The `BaseFakeFPQuantizeFunction` class is a PyTorch function that performs fake FP8 quantization. It takes in an input
tensor, step size, zero point, quantization minimum, quantization maximum, and axis as inputs. The `symbolic` method
defines the symbolic computation graph for the function, which checks if the quantization minimum and maximum are valid
for FP8 quantization. If valid, it calls the `fp8_qdq_symbolic` function to perform the quantization and dequantization.

The `fake_fp8_quantize` function performs the actual fake FP8 quantization. It takes in an input tensor, step size,
zero point, quantization minimum, quantization maximum, and axis as inputs. It first adjusts the step size and zero
point according to the axis, then performs the quantization by dividing the input by the step size and adding the zero
point. The result is then clipped to the quantization minimum and maximum, and converted to the FP8 data type.
Finally, the result is converted back to the original data type, subtracted by the zero point, and multiplied by the
step size.

The `fp8_qdq_symbolic` function defines the symbolic computation graph for the fake FP8 quantization. It takes in
an input value, step size, zero point, and axis as inputs. It first casts the zero point to the FP8 data type, then
performs the quantization using the `QuantizeLinear` operator. The result is then dequantized using the
`DequantizeLinear` operator.

Note:
    This implementation assumes that the quantization minimum and maximum are valid for FP8 quantization.
    It also assumes that the input tensor is a PyTorch tensor.
"""

# pylint: disable=unused-argument
import torch
from torch import Tensor, Value
from torch._C._onnx import TensorProtoDataType
from torch.autograd import Function
from torch.onnx._internal import jit_utils


# pylint: disable-next=abstract-method
class BaseFakeFPQuantizeFunction(Function):
    """An autograd function that performs fake FP quantization.

    Static Methods:
        symbolic: Defines the symbolic computation graph for the function.
    """

    @staticmethod
    @torch.onnx.symbolic_helper.parse_args("v", "v", "v", "none", "none", "none", "i")  # type: ignore # pylint: disable-next=too-many-positional-arguments
    def symbolic(
        g: jit_utils.GraphContext,
        inputs: Value,
        step_size: Value,
        zero_point: Value,
        grad_scale: float,
        quant_min: float,
        quant_max: float,
        axis: int | None,
    ) -> Value | tuple[Value, ...]:
        r"""Define the symbolic computation graph for fake FP8 quantization.

        Args:
            g (`jit_utils.GraphContext`): The graph context.
            inputs (`torch.Value`): A tensor to quantize.
            step_size (`torch.Value`): The quantization scale, determining the magnitude of each quantization interval.
            zero_point (`torch.Tensor`): The quantization zero\_point. It may be expressed as a float in the context of
                                        asymmetric quantization, while for symmetric quantization, it is fixed at 0.
            grad_scale (`float`): The gradient scale.
            quant_min (`float`): The lower bound of the quantized domain, specified as an integer.
            quant_max (`float`): The upper bound of the quantized domain in as an integer.
            axis (`int`, optional): Channel axis. Only used when `per_channel` is `True`. Defaults to 0.

        Returns:
            The output value.
        """
        if (quant_min, quant_max) != (torch.finfo(torch.float8_e4m3fn).min, torch.finfo(torch.float8_e4m3fn).max):
            raise torch.onnx.errors.SymbolicValueError(
                "For fp quantizer's (quant_min, quant_max), ONNX allows only "
                f"({torch.finfo(torch.float8_e4m3fn).min}, {torch.finfo(torch.float8_e4m3fn).max}). "
                f"Got ({quant_min}, {quant_max})",
                inputs,
            )
        zero_point = g.op("Cast", zero_point, to_i=TensorProtoDataType.FLOAT8E4M3FN)
        quantized = g.op("QuantizeLinear", inputs, step_size, zero_point, axis_i=axis)
        dequantized = g.op("DequantizeLinear", quantized, step_size, zero_point, axis_i=axis)
        return dequantized


def fake_fp8_quantize(
    inputs: Tensor,
    step_size: Tensor,
    zero_point: Tensor,
    *,
    quant_min: float,
    quant_max: float,
    axis: int | None = None,
) -> torch.Tensor:
    """Perform fake FP8 quantization on an input tensor.

    Args:
        inputs (`torch.Tensor`): The input tensor.
        step_size (`torch.Tensor`): The step size.
        zero_point (`torch.Tensor`): The zero point.
        quant_min (`float`): The quantization minimum.
        quant_max (`float`): The quantization maximum.
        axis (`int`, optional): The axis.

    Returns:
        Value | tuple[Value, ...]: The quantized tensor.
    """
    if axis is not None:
        dimlist = [1] * inputs.dim()
        dimlist[axis] = -1
        step_size = step_size.reshape(dimlist)
        zero_point = zero_point.reshape(dimlist)
    out = (inputs / step_size) + zero_point
    out = out.clip(quant_min, quant_max)
    out = out.to(dtype=torch.float8_e4m3fn)
    out = out.to(dtype=inputs.dtype) - zero_point
    out = out * step_size
    return out

from typing import Callable, Optional

import torch
from torch import Tensor


def fake_quantize(
    inputs: Tensor,
    step_size: Tensor,
    zero_point: Tensor,
    quant_min: int,
    quant_max: int,
    axis: Optional[int] = None,
) -> torch.Tensor:
    r"""Same as `torch.fake_quantize_per_channel_affine` if `per_channel` is `True`, otherwise
    `torch.fake_quantize_per_tensor_affine`

    In OwLite, quantization is simulated through the following mathematical expression:

    $$
    \small

    \text{FakeQuantize}(\text{input})= \text{clip} \left( {\lfloor \frac{\text{input} - \text{zero\_point}}{\text
    {step\_size}} \rceil }, \text{quant\_min}, \text{quant\_max} \right) \cdot \text{step\_size} + \text{zero\_point}

    $$

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
    if axis is not None:
        return torch.fake_quantize_per_channel_affine(
            inputs,
            step_size,
            zero_point,
            axis,
            quant_min,
            quant_max,
        )

    return torch.fake_quantize_per_tensor_affine(
        inputs,
        step_size,
        # `torch.fake_quantize_per_tensor_affine` expects `zero_point` to be either int32 or int64
        # (See https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_tensor_affine.html)
        # while `torch.fake_quantize_per_channel_affine` doesn't
        zero_point,
        quant_min=quant_min,
        quant_max=quant_max,
    )


FakeQuantizeSignature = Callable[
    [
        Tensor,  # inputs
        Tensor,  # step_size
        Tensor,  # zp
        float,  # grad_scale
        int,  # quant_min
        int,  # quant_max
        Optional[int],  # axis
        bool,  # compensate_zp
    ],
    Tensor,
]

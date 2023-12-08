"""Fake quantization using configuration"""
from typing import Callable, Union

import torch
from torch import IntTensor, Tensor


def fake_quantize(
    inputs: Union[Tensor, float],
    step_size: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: Union[IntTensor, int],
    quant_max: Union[IntTensor, int],
    per_channel: Union[torch.BoolTensor, bool],
    axis: int = 0,
) -> torch.Tensor:
    """Same as `torch.fake_quantize_per_channel_affine` if `per_channel` is `True`, otherwise
    `torch.fake_quantize_per_tensor_affine`

    Args:
        inputs (torch.Tensor): A tensor to quantize.
        step_size (torch.Tensor): A float tensor which is quantization scales.
        zero_point (torch.Tensor): A float tensor, quantization zero_point.
        quant_min (int): The lower bound of the quantized domain.
        quant_max (int): The upper bound of the quantized domain.
        per_channel (bool): If True, input will be per-channel quantized, otherwise per-tensor quantized.
        axis (int, optional): Channel axis. Only used when `per_channel` is `True`. Defaults to 0.

    Returns:
        torch.Tensor: fake-quantized tensor
    """
    if per_channel:
        return torch.fake_quantize_per_channel_affine(
            inputs,
            step_size,
            zero_point,
            axis,
            quant_min=quant_min,
            quant_max=quant_max,
        )

    return torch.fake_quantize_per_tensor_affine(
        inputs,
        step_size,
        # `torch.fake_quantize_per_tensor_affine` expects `zero_point` to be either int32 or int64
        # pylint: disable-next=line-too-long
        # (See https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_tensor_affine.html#torch-fake-quantize-per-tensor-affine)
        # while `torch.fake_quantize_per_channel_affine` doesn't
        zero_point.to(torch.int32).to(zero_point.device),
        quant_min=quant_min,
        quant_max=quant_max,
    )


FakeQuantFunc = Callable[
    [
        Union[Tensor, float],  # inputs
        Tensor,  # step_size
        Tensor,  # zp
        Union[Tensor, float],  # grad_scale
        Union[IntTensor, int],  # quant_min
        Union[IntTensor, int],  # quant_max
        Union[torch.BoolTensor, bool],  # per_channel
        bool,  # compensate_zp
    ],
    Tensor,
]

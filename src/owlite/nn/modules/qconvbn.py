import torch
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d, _BatchNorm

from .fake_quantizer import FakeQuantizer
from .qconv import QConv1d, QConv2d, QConv3d, _QConvNd


class _QConvBnNd(torch.nn.Module):
    """Base class of quantized covolution with followed batchnorm layer."""

    def __init__(self, qconv: _QConvNd, bn: _BatchNorm):
        super().__init__()
        self.qconv: _QConvNd = qconv
        self.bn: _BatchNorm | None = bn

    # pylint: disable=protected-access
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward with quantized convolution with batch normalization."""
        if self.bn is None:
            return self.qconv(inputs)
        conv_output = self.qconv._conv_forward(inputs, self.qconv._get_weight(), self.qconv.bias)
        bn_output = self.bn(conv_output)
        bn_output = self._folding_forward(bn_output)
        return bn_output

    # pylint: enable=protected-access
    def _folding_forward(self, inputs: Tensor) -> Tensor:
        if (
            self.bn is None
            or not self.qconv.int32_bias
            or not FakeQuantizer.check_if_enabled(self.qconv.weight_quantizer)
            or not FakeQuantizer.check_if_enabled(self.qconv.input_quantizer)
        ):
            return inputs
        assert self.bn.running_mean is not None
        assert self.bn.running_var is not None
        # assume that inputs shape is (n,cin,*) and weight shape is (cout,cin,*kernel_shape)
        weight_shape = [1] * inputs.dim()
        weight_shape[1] = -1
        channel_dims = list(range(inputs.dim()))
        channel_dims.pop(1)
        qconv_bias = self.qconv.bias if self.qconv.bias is not None else torch.zeros_like(self.bn.bias)
        mean = inputs.mean(dim=channel_dims) if self.bn.training else self.bn.running_mean
        var = inputs.std(dim=channel_dims) ** 2 if self.bn.training else self.bn.running_var
        alpha = torch.rsqrt(var + self.bn.eps) * self.bn.weight
        beta = self.bn.bias - self.bn.weight * mean * torch.rsqrt(var + self.bn.eps)
        fused_scale = alpha * self.qconv.weight_quantizer.step_size * self.qconv.input_quantizer.step_size
        fused_bias = (alpha * qconv_bias + beta) / fused_scale
        output = inputs - ((fused_bias - fused_bias.round()) * (fused_scale)).detach().reshape(weight_shape)
        return output


class QConvBn1d(_QConvBnNd):
    """This module sequentially calls the `QConv1d` and `BatchNorm1d` modules if they are available."""

    def __init__(self, qconv: QConv1d, bn: BatchNorm1d):
        super().__init__(qconv, bn)
        self.qconv: QConv1d
        self.bn: BatchNorm1d | None


class QConvBn2d(_QConvBnNd):
    """This module sequentially calls the `QConv2d` and `BatchNorm2d` modules if they are available."""

    def __init__(self, qconv: QConv2d, bn: BatchNorm2d):
        super().__init__(qconv, bn)
        self.qconv: QConv2d
        self.bn: BatchNorm2d | None


class QConvBn3d(_QConvBnNd):
    """This module sequentially calls the `QConv3d` and `BatchNorm3d` modules if they are available."""

    def __init__(self, qconv: QConv3d, bn: BatchNorm3d):
        super().__init__(qconv, bn)
        self.qconv: QConv3d
        self.bn: BatchNorm3d | None

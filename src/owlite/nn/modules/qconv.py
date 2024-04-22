import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules.conv import Conv1d, Conv2d, Conv3d, _ConvNd

from ...options import Channel, FakeQuantizerOptions
from ..functions.ste import scaled_round_ste
from .fake_quantizer import FakeQuantizer
from .qmodule_mixins import UnaryNeuralQModuleMixin


# mypy: disable-error-code=misc
class _QConvNd(_ConvNd, UnaryNeuralQModuleMixin):
    """Base class for quantized convolution layer inherited from [torch.nn.modules.conv._ConvNd](https://github.com/pytorch/pytorch/blob/4c55dc50355d5e923642c59ad2a23d6ad54711e7/torch/nn/modules/conv.py).

    It performs convolution operations using the input and fake-quantized weights. Its weights and biases are copied
    from the original convolution instance.

    ### Attributes
    - __input_quantizer__ (`FakeQuantizer`): fake quantizer used for the input.
    - __weight_quantizer__ (`FakeQuantizer`): fake quantizer used for the weights.
    # `class owlite.nn.Qconv1d`
    Quantized 1D convolution module
    # `class owlite.nn.Qconv2d`
    Quantized 2D convolution module
    # `class owlite.nn.Qconv3d`
    Quantized 3D convolution module

    Args:
        _ConvNd (_type_): _description_
        UnaryNeuralQModuleMixin (_type_): _description_
    """

    def __init__(
        self,
        conv: _ConvNd,
        weight_options: FakeQuantizerOptions | None = None,
    ):
        # pylint: disable=no-value-for-parameter
        super().__init__(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,  # type: ignore[arg-type]
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )  # type: ignore
        self.train(conv.training)
        self.input_quantizer: FakeQuantizer | None = None
        self.int32_bias: bool = False
        channel = (
            Channel(axis=0, size=self.out_channels)
            if (weight_options is not None and weight_options.per_channel)
            else None
        )
        self.weight_quantizer = FakeQuantizer.create(weight_options, channel, narrow_range=True)
        if self.weight_quantizer is not None:
            self.weight_quantizer.to(self.weight.device)
        with torch.no_grad():
            self.weight.copy_(conv.weight)
            if self.bias is not None and conv.bias is not None:
                self.bias.copy_(conv.bias)

    def _set_bias_to_zero(self) -> None:
        self.bias = Parameter(torch.zeros(self.out_channels).to(self.weight.device))

    def _get_weight(self) -> Tensor:
        if self.weight_quantizer is None:
            return self.weight
        return self.weight_quantizer(self.weight)

    def _get_bias(self) -> Tensor | None:
        if (
            self.bias is None
            or not self.int32_bias
            or not FakeQuantizer.check_if_enabled(self.weight_quantizer)
            or not FakeQuantizer.check_if_enabled(self.input_quantizer)
        ):
            return self.bias
        scale = self.weight_quantizer.step_size.detach() * self.input_quantizer.step_size.detach()
        return scaled_round_ste(self.bias, scale)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward with quantized weight if available."""
        return self._conv_forward(inputs, self._get_weight(), self._get_bias())

    def extra_repr(self) -> str:
        extra_repr = super().extra_repr()
        if self.int32_bias:
            extra_repr = f"{extra_repr}, int32"
        return extra_repr


class QConv1d(_QConvNd, Conv1d):
    """Applies a 1D convolution over an input signal composed of several input planes with quantized weight."""

    def __init__(self, conv: Conv1d, weight_options: FakeQuantizerOptions | None = None) -> None:
        """Convert a `Conv1d` instance to the analogous `QConv1d` instance, copying weights and bias if exists.

        Args:
            conv (torch.nn.Conv1d): a `Conv1d` instance to be converted to `QConv1d` instance.
            weight_options (FakeQuantizerOptions | None, optional): Option for the weight fake quantizer.
                Defaults to None.
        """
        super().__init__(conv, weight_options)


class QConv2d(_QConvNd, Conv2d):
    """Applies a 2D convolution over an input signal composed of several input planes with quantized weight."""

    def __init__(self, conv: Conv2d, weight_options: FakeQuantizerOptions | None = None) -> None:
        """Convert a `Conv2d` instance to the analogous `QConv2d` instance, copying weights and bias if exists.

        Args:
            conv (torch.nn.Conv2d): a `Conv2d` object to be converted to `QConv2d` instance.
            weight_options (FakeQuantizerOptions | None, optional): instance for the weight fake quantizer.
                Defaults to None.
        """
        super().__init__(conv, weight_options)


class QConv3d(_QConvNd, Conv3d):
    """Applies a 3D convolution over an input signal composed of several input planes with quantized weight."""

    def __init__(self, conv: Conv3d, weight_options: FakeQuantizerOptions | None = None) -> None:
        """Convert a `Conv3d` instance to the analogous `QConv3d` instance, copying weights and bias if exists.

        Args:
            conv (torch.nn.Conv3d): a `Conv3d` instance to be converted to `QConv3d` instance.
            weight_options (FakeQuantizerOptions | None, optional): Option for the weight fake quantizer.
                Defaults to None.
        """
        super().__init__(conv, weight_options)

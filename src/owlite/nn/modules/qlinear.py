# pylint: disable=not-callable
import torch
import torch.nn.functional as F
from torch import Tensor

from ...options import Channel, FakeQuantizerOptions
from .fake_quantizer import FakeQuantizer
from .qmodule_mixins import UnaryNeuralQModuleMixin


# mypy: disable-error-code=misc
class QLinear(torch.nn.Linear, UnaryNeuralQModuleMixin):
    r"""Applies a linear transformation with fake-quantized weight $$ A_q $$ to the incoming data: $$ y = xA_q^T + b $$.

    Additionally, fake-quantization is applicable to both the bias and bias addition:
    $$y = \text{quant}(xW_q^T) + \text{quant}(b)$$, where represents $$\text{quant}$$ the fake-quantize function.
    The module copies the weights and biases from the original linear instance.

    Quantized linear layer inherited from
    [torch.nn.Linear](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py).
    """

    def __init__(
        self,
        linear: torch.nn.Linear,
        weight_opts: FakeQuantizerOptions | None = None,
    ) -> None:
        """Convert a `Linear` instance to the analogous `QLinear` instance, copying weights and bias if exists.

        Args:
            linear (`torch.nn.Linear`): a `Linear` instance to be converted to `QLinear` instance.
            weight_opts (`FakeQuantizerOptions | None`, optional): Option for the fake weight quantizer.
                Defaults to None.
        """
        super().__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype,
        )
        self.train(linear.training)
        self.input_quantizer: FakeQuantizer | None = None
        with torch.no_grad():
            self.weight.copy_(linear.weight)
            if self.bias is not None:
                self.bias.copy_(linear.bias)
        channel = (
            Channel(axis=0, size=self.out_features) if (weight_opts is not None and weight_opts.per_channel) else None
        )
        self.weight_quantizer = FakeQuantizer.create(weight_opts, channel, narrow_range=True)
        if self.weight_quantizer is not None:
            self.weight_quantizer.to(self.weight.device)
        self.bias_quantizer: FakeQuantizer | None = None
        self.hidden_input_quantizer: FakeQuantizer | None = None

    def _set_bias_to_zero(self) -> None:
        self.bias = torch.nn.Parameter(torch.zeros(self.out_features).to(self.weight.device))

    # pylint: disable=arguments-renamed, invalid-name
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward with quantized weight if available."""
        weight = self.weight_quantizer(self.weight) if self.weight_quantizer is not None else self.weight
        bias = (
            self.bias_quantizer(self.bias) if self.bias_quantizer is not None and self.bias is not None else self.bias
        )

        if self.hidden_input_quantizer is not None and bias is not None:
            x = F.linear(inputs, weight, None)
            x = self.hidden_input_quantizer(x)
            return torch.add(bias, x)
        return F.linear(inputs, weight, bias)

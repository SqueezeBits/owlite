# pylint: disable=not-callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ...options import Channel, FakeQuantizerOptions
from .fake_quantizer import FakeQuantizer
from .qmodule_mixins import UnaryNeuralQModuleMixin


# mypy: disable-error-code=misc
class QLinear(torch.nn.Linear, UnaryNeuralQModuleMixin):
    """Applies a linear transformation to the incoming data: :math:`y = xA_q^T + b`,
    where :math:`A_q` represents the fake-quantized weight.
    """

    def __init__(
        self,
        linear: torch.nn.Linear,
        weight_opts: Optional[FakeQuantizerOptions] = None,
    ) -> None:
        """Initializes instance from an existing `torch.nn.Linear` instance, copying the weights and bias if it exists.

        Args:
            linear (torch.nn.Linear): An existing `torch.nn.Linear` instance.
            weight_opts (Optional[FakeQuantizerOptions], optional): Option for the fake weight quantizer.
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
        self.input_quantizer: Optional[FakeQuantizer] = None
        with torch.no_grad():
            self.weight.copy_(linear.weight)
            if self.bias is not None:
                self.bias.copy_(linear.bias)
        channel = Channel(0, self.out_features) if (weight_opts is not None and weight_opts.per_channel) else None
        self.weight_quantizer = FakeQuantizer.create(weight_opts, channel, narrow_range=True)
        if self.weight_quantizer is not None:
            self.weight_quantizer.to(self.weight.device)
        self.bias_quantizer: Optional[FakeQuantizer] = None
        self.hidden_input_quantizer: Optional[FakeQuantizer] = None

    def _set_bias_to_zero(self) -> None:
        self.bias = torch.nn.Parameter(torch.zeros(self.out_features).to(self.weight.device))

    # pylint: disable=arguments-renamed, invalid-name
    def forward(self, inputs: Tensor) -> Tensor:
        """Quantized linear forward"""
        weight = self.weight_quantizer(self.weight) if self.weight_quantizer is not None else self.weight
        bias = (
            self.bias_quantizer(self.bias) if self.bias_quantizer is not None and self.bias is not None else self.bias
        )

        if self.hidden_input_quantizer is not None and bias is not None:
            x = F.linear(inputs, weight, None)
            x = self.hidden_input_quantizer(x)
            return torch.add(bias, x)
        return F.linear(inputs, weight, bias)

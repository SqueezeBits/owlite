from abc import ABC, abstractmethod

import torch

from ...owlite_core.logger import log
from .fake_quantizer import FakeQuantizer


class UnaryNeuralQModuleMixin(ABC):
    """Mixin-class for fake-quantized counterparts of subclasses of `torch.nn.Module`.

    This mixin assumes that the base class has parameters named `weight` and `bias`.
    and that its `forward` method takes exactly one parameter other than `self`.
    Examples: `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, `torch.nn.Linear`.
    """

    weight: torch.nn.Parameter
    bias: torch.nn.Parameter | None
    input_quantizer: FakeQuantizer | None
    weight_quantizer: FakeQuantizer | None

    @abstractmethod
    def _set_bias_to_zero(self) -> None:
        pass

    def clip_weight(self) -> None:
        """Clip the weights with narrow range.

        If the weight quantizer exists and narrow range is True, clip the weight values to fit the narrow range.
        Otherwise, do nothing.

        Raises:
            RuntimeError: in tracing.
        """
        if torch.jit.is_tracing():
            log.error("Trying to clipping range a module in tracing(torch.jit.trace)")
            log.error(self)
            raise RuntimeError("Trying to clipping range a module in tracing(torch.jit.trace)")
        if self.weight_quantizer is None:
            return
        if not self.weight_quantizer.narrow_range:
            log.debug("Trying to clipping range a module with the weight quantizer that is not a narrow range.")
            log.debug(self.weight_quantizer)
            return
        # convert all step_size to be positive
        self.weight_quantizer.step_size.data = self.weight_quantizer.step_size.data.abs()

        clip_min_values = (self.weight_quantizer.quant_min) * self.weight_quantizer.step_size.data
        clip_max_values = (self.weight_quantizer.quant_max) * self.weight_quantizer.step_size.data
        shape = [-1, *[1 for _ in range(self.weight.dim() - 1)]]
        self.weight.data = self.weight.data.clip(clip_min_values.reshape(shape), clip_max_values.reshape(shape))
        return

    def enable(self) -> None:
        """Enable the input and weight quantizers."""
        if self.input_quantizer is not None:
            self.input_quantizer.enable()
        if self.weight_quantizer is not None:
            self.weight_quantizer.enable()

    def disable(self) -> None:
        """Disable the input and weight quantizers."""
        if self.input_quantizer is not None:
            self.input_quantizer.disable()
        if self.weight_quantizer is not None:
            self.weight_quantizer.disable()

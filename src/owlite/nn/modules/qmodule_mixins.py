from typing import Optional

import torch

from ...owlite_core.logger import log
from .fake_quantizer import FakeQuantizer


class UnaryNeuralQModuleMixin:
    """
    Mixin-class for implementing weight-quantized counterparts of subclasses
    of torch.nn.Module with the parameters named 'weight' and 'bias'
    such that whose `forward` method takes exactly one parameter other than 'self'.
    Examples: `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, `torch.nn.Linear`
    """

    weight: torch.nn.Parameter
    bias: Optional[torch.nn.Parameter]
    input_quantizer: Optional[FakeQuantizer]
    weight_quantizer: Optional[FakeQuantizer]

    def _set_bias_to_zero(self) -> None:
        """makes bias with zero tensor"""
        raise NotImplementedError()

    def check_folding_condition(self) -> bool:
        """checks quantized class can folding in bias.

        Verify that input and weight quantizers satisfy the folding condition.
        Input quantization should be asymmetric, and weight quantization should be symmetric.

        Returns:
            bool: True if it satisfy the folding condition. Otherwise, False.
        """
        if self.input_quantizer is None or self.weight_quantizer is None:
            return False
        if self.input_quantizer.symmetric:
            log.debug_warning(
                f"Trying to folding zero point to bias though symmetric quantization({self.input_quantizer})",
                stacklevel=2,
            )
            return False
        if self.input_quantizer.precision >= 16:
            log.debug_warning(
                "Trying to folding zero point to bias "
                f"though input precision is FP16 or FP32({self.input_quantizer})",
                stacklevel=2,
            )
            return False
        if not self.weight_quantizer.symmetric:
            raise RuntimeError("Asymmetric weight quantization is not supported")
        return True

    def clip_weight(self) -> None:
        """Clips the weights with narrow range.

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

    def fold_input_quantizer_zero_point_to_bias(self) -> None:
        """Folds precomputing quantized term to bias"""
        if self.input_quantizer is None:
            return
        if not self.check_folding_condition():
            return
        if self.input_quantizer.is_zero_point_folded:
            log.debug_warning(
                f"Trying to folding zero point to bias though it's already folded({self.input_quantizer})",
                stacklevel=2,
            )
            return

        # initialize bias if it is not existed
        if self.bias is None:
            self._set_bias_to_zero()
        weight_dim = list(range(1, self.weight.dim()))  # assume channel dim is 0
        precomputed = self.weight.data.sum(dim=weight_dim) * self.input_quantizer.zero_point
        if self.bias is not None:
            self.bias.data = self.bias.data + precomputed
        self.input_quantizer.is_zero_point_folded = True

    def unfold_input_quantizer_zero_point_to_bias(self) -> None:
        """Unfolds precomputing quantized term from bias"""
        if self.input_quantizer is None or self.weight_quantizer is None:
            return
        if not self.check_folding_condition():
            return
        if self.bias is None:
            log.error("Trying to folding zero point to bias with no bias module.")
            log.error(f"{self}")
            raise ValueError("Trying to folding zero point to bias with no bias module.")
        if not self.input_quantizer.is_zero_point_folded:
            log.warning(
                f"Trying to folding zero point to bias though it's already folded({self.input_quantizer})",
                stacklevel=2,
            )
            return
        with torch.no_grad():
            non_channel_dims = list(range(1, self.weight.dim()))  # assume channel dim is 0
            precomputed = (
                self.weight_quantizer(self.weight.data).sum(dim=non_channel_dims) * self.input_quantizer.zero_point
            )
            self.bias.data = self.bias.data - precomputed
        self.input_quantizer.is_zero_point_folded = False

    def enable(self) -> None:
        """Enables the quantizer."""
        if self.input_quantizer is not None:
            self.input_quantizer.enable()
        if self.weight_quantizer is not None:
            self.weight_quantizer.enable()

    def disable(self) -> None:
        """Disables quantizers"""
        if self.input_quantizer is not None:
            self.input_quantizer.disable()
        if self.weight_quantizer is not None:
            self.weight_quantizer.disable()

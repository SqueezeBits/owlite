from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from ..core.logger import log

if TYPE_CHECKING:
    from ..nn import FakeQuantizer


class Calibrator(ABC):
    """Base calibrator abstract class.

    Uses the forward hook to collect the data needed for calibration and update the quantizer's
    step_size and zero_point.

    In **OwLite**, calibrator classes collect the necessary data for calibration based on the data passing through
    the `FakeQuantizer`. This process enables the determination of the `FakeQuantizer`'s `step_size` and `zero_point`.

    Attributes:
        hook_handler (`torch.utils.hooks.RemovableHandle`, `optional`): A hook handler.
        quantizer (`FakeQuantizer`): The `FakeQuantizer` to which the calibration will be applied.
    """

    def __init__(self, quantizer: "FakeQuantizer"):
        self.hook_handler: RemovableHandle | None = None
        self.quantizer: FakeQuantizer = quantizer

    def check_calib_ready(self) -> bool:
        """Check that the conditions for calibration are met.

        Returns:
            `True`, if all conditions for calibration are met, `False` otherwise.
        """
        if self.quantizer.calibrator is not self:
            log.error("The calibrator does not match the calibrator that the quantizer points")
            return False
        if self.quantizer.is_enabled:
            log.error("The quantizer should be disabled during calibration.")
            return False
        return True

    def update_fake_quantizer_param_with_max_min(self, max_value: Tensor, min_value: Tensor | None = None) -> None:
        """Find and apply the step_size and zero_points of a quantizer with the given values as min and max.

        Each parameters are updated as follows(For symmetric quantization, min_value only uses 0).

        step_size = (max_value - min_value) / (quant_max - quant_min)
        zero_point = - round(min_value / step_size) + quant_min

        Args:
            max_value(Tensor): The maximum value that will not be clipped.
            min_value(Tensor | None): The minimum value that will not be clipped. This value will only be
                used for asymmetric quantization. Defaults to None.

        Raises:
            TypeError: When the shape of the fake quantizer's parameters and arguments do not match.
            TypeError: When min_value is not specified in asymmetric quantization
        """
        if self.quantizer.step_size.shape != max_value.shape:
            raise TypeError(
                f"Tensor shape of step_size({self.quantizer.step_size.shape}) is not matched to"
                f"max_value({max_value.shape})"
            )
        if self.quantizer.symmetric:
            self.quantizer.step_size.data = (max_value / self.quantizer.maxabs_bound).detach().clone()
            return
        if min_value is None:
            raise TypeError("Trying to update the asymmetric quantizer parameters, but no min_value was given.")
        if self.quantizer.step_size.shape != min_value.shape:
            raise TypeError(
                f"Tensor shape of step_size({self.quantizer.step_size.shape}) is not matched to"
                f"min_value({min_value.shape})"
            )
        max_value = torch.where(max_value >= 0, max_value, 0.0).clone()
        min_value = torch.where(min_value <= 0, min_value, 0.0).clone()
        self.quantizer.step_size.data = (
            ((max_value - min_value) / (self.quantizer.quant_max - self.quantizer.quant_min)).detach().clone()
        )
        self.quantizer.zero_point.data = (
            (-(min_value / self.quantizer.step_size.data).round() + self.quantizer.quant_min)
            .to(self.quantizer.zero_point.data.dtype)
            .detach()
            .clone()
        )

    @abstractmethod
    def prepare(self) -> RemovableHandle:
        """Prepare calibration for the quantizer.

        Set temporal attributes on the quantizer and register a hook on the quantizer.

        Raises:
            ValueError: If the attributions are already set.

        Returns:
            torch.utils.hooks.RemovableHandle: A registered hook handler.
        """

    @abstractmethod
    def update(self) -> None:
        """Calculate step_size and zero_point of quantizer and update them. Then remove the registered hook."""

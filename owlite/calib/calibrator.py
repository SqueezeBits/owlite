from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from torch.utils.hooks import RemovableHandle

from owlite_core.logger import log

if TYPE_CHECKING:
    from ..nn.fake_quantizer import FakeQuantizer


class Calibrator(ABC):
    """Base calibrator abstract class

    Uses the forward hook to collect the data needed for calibration and update the quantizer's
    step_size and zero_point.

    Attributes:
        hook_handler (Optional[torch.utils.hooks.RemovableHandle]): A hook handler.
        quantizer (FakeQuantizer): The `FakeQuantizer` to which the calibration will be applied.
    """

    def __init__(self, quantizer: "FakeQuantizer"):
        self.hook_handler: Optional[RemovableHandle] = None
        self.quantizer: FakeQuantizer = quantizer

    def check_calib_ready(self) -> bool:
        """checks that the conditions for calibration are met

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

    @abstractmethod
    def prepare(self) -> RemovableHandle:
        """Prepares calibration for the quantizer.

        Set temporal attributes on the quantizer and register a hook on the quantizer.

        Raises:
            ValueError: If the attributions are already set.

        Returns:
            torch.utils.hooks.RemovableHandle: A registered hook handler.
        """

    @abstractmethod
    def update(self) -> None:
        """Calculate step_size and zero_point of quantizer and update them. Then remove the registered hook."""

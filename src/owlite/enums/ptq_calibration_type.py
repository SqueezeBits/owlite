from enum import IntEnum
from typing import TYPE_CHECKING

from ..owlite_core.logger import log

if TYPE_CHECKING:
    from ..calib.calibrator import Calibrator


# pylint: disable=invalid-name
class PTQCalibrationType(IntEnum):
    """The enum for specifying available Calibrator classes."""

    absmax = 0
    percentile = 1
    mse = 2
    minmax = 3
    entropy = 4

    @property
    def calibrator_class(self) -> type["Calibrator"]:
        """The Calibrator class corresponding to this enum value."""
        # pylint: disable-next=import-outside-toplevel
        from ..calib import (
            EntropyCalibrator,
            MinmaxCalibrator,
            MSECalibrator,
            PercentileCalibrator,
        )

        predefined_classes: dict[str, type[Calibrator]] = {
            "absmax": MinmaxCalibrator,
            "percentile": PercentileCalibrator,
            "mse": MSECalibrator,
            "minmax": MinmaxCalibrator,
            "entropy": EntropyCalibrator,
        }
        if self.name == "absmax":
            log.warning("`absmax` is deprecated and will be removed in the future release. Use `minmax` instead.")  # UX
        return predefined_classes[self.name]

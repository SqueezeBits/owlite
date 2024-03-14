from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..calib.calibrator import Calibrator


# pylint: disable=invalid-name
class PTQCalibrationType(Enum):
    """The enum for specifying available Calibrator classes"""

    absmax = 0
    percentile = 1
    mse = 2
    minmax = 3
    entropy = 4

    @property
    def calibrator_class(self) -> type["Calibrator"]:
        """The Calibrator class corresponding to this enum value"""
        # pylint: disable-next=import-outside-toplevel
        from ..calib import (
            AbsmaxCalibrator,
            EntropyCalibrator,
            MinmaxCalibrator,
            MSECalibrator,
            PercentileCalibrator,
        )

        predefined_classes: dict[str, type[Calibrator]] = {
            "absmax": AbsmaxCalibrator,
            "percentile": PercentileCalibrator,
            "mse": MSECalibrator,
            "minmax": MinmaxCalibrator,
            "entropy": EntropyCalibrator,
        }
        return predefined_classes[self.name]

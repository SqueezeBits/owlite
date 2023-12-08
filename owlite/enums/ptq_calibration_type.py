"""The enumeration of available Calibrator implementations"""
from enum import Enum

from ..calib import (
    AbsmaxCalibrator,
    MinmaxCalibrator,
    MSECalibrator,
    PercentileCalibrator,
)


# pylint: disable=invalid-name
class PTQCalibrationType(Enum):
    """The enum for specifying available Calibrator classes"""

    absmax = 0
    percentile = 1
    mse = 2
    minmax = 3

    @property
    def calibrator_class(self) -> type:
        """The Calibrator class corresponding to this enum value"""
        return {
            "absmax": AbsmaxCalibrator,
            "percentile": PercentileCalibrator,
            "mse": MSECalibrator,
            "minmax": MinmaxCalibrator,
        }[self.name]

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

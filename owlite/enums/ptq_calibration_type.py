from enum import Enum


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
        # pylint: disable-next=import-outside-toplevel
        from ..calib import (
            AbsmaxCalibrator,
            MinmaxCalibrator,
            MSECalibrator,
            PercentileCalibrator,
        )

        return {
            "absmax": AbsmaxCalibrator,
            "percentile": PercentileCalibrator,
            "mse": MSECalibrator,
            "minmax": MinmaxCalibrator,
        }[self.name]

from dataclasses import dataclass, field
from typing import Any, Optional

from typing_extensions import Self

from ..enums.ptq_calibration_type import PTQCalibrationType
from ..enums.qat_backward_type import QATBackwardType
from ..owlite_core.logger import log, suppress_owlite_warnings
from .options_mixin import OptionsMixin


# pylint: disable=too-many-instance-attributes
@suppress_owlite_warnings
@dataclass
class FakeQuantizerOptions(OptionsMixin):
    """Options required for setting up a quantizer"""

    qat_backward: QATBackwardType = QATBackwardType.ste
    ptq_calibration: PTQCalibrationType = PTQCalibrationType.absmax
    precision: int = field(default=8)
    per_channel: bool = field(default=False)
    symmetric: bool = field(default=True)
    learn_zero_point: bool = field(default=False)
    unsigned: bool = field(default=False)
    grad_scale: float = field(default=1.000)
    percentile: Optional[float] = field(default=None)

    @classmethod
    def ste_per_channel(cls, **kwargs: Any) -> Self:
        """A convenience wrapper for creating options with "ste" backward and per channel quantization"""
        return cls(qat_backward=QATBackwardType.ste, per_channel=True, **kwargs)

    @classmethod
    def ste_per_tensor(cls, **kwargs: Any) -> Self:
        """A convenience wrapper for creating options with "ste" backward and per tensor quantization"""
        return cls(qat_backward=QATBackwardType.ste, per_channel=False, **kwargs)

    @classmethod
    def clq_per_channel(cls, **kwargs: Any) -> Self:
        """A convenience wrapper for creating options with "clq" backward and per channel quantization"""
        return cls(qat_backward=QATBackwardType.clq, per_channel=True, **kwargs)

    @classmethod
    def clq_per_tensor(cls, **kwargs: Any) -> Self:
        """A convenience wrapper for creating options with "clq" backward and per tensor quantization"""
        return cls(qat_backward=QATBackwardType.clq, per_channel=False, **kwargs)

    def check_precision(self, precision: int) -> bool:
        """precision must be one of 4, 8 or 16"""
        return precision in (4, 8, 16)

    def check_percentile(self, percentile: Optional[float]) -> bool:
        """
        if `ptq_calibration="percentile"`, `percentile` value must be provided
        and it must be strictly greater than 0 and less than or equal to 100.
        Otherwise, its value is ignored.
        """
        if self.ptq_calibration == PTQCalibrationType.percentile:
            return percentile is not None and 0 < percentile <= 100
        if percentile is not None:
            log.warning(
                '`percentile` is used only when `ptq_calibration="percentile"`.'
                f"The given percentile value {percentile} will be ignored",
                stacklevel=2,
            )
        return True

    def check_grad_scale(self, grad_scale: float) -> bool:
        """grad_scale value must be between 0 and 1 (inclusive)"""
        return 0 <= grad_scale <= 1

    def check_learn_zero_point(self, learn_zero_point: bool) -> bool:
        """`learn_zero_point` must be False if `symmetric=True`"""
        return not (self.symmetric and learn_zero_point)

    def check_per_channel(self, per_channel: bool) -> bool:
        """`per_channel=True` is not compatible with `symmetric=False`"""
        return self.symmetric or not per_channel

    def check_symmetric(self, symmetric: bool) -> bool:
        """
        * `learn_zero_point` must be False if `symmetric=True`
        * `ptq_calibration="absmax"` is not compatible with `symmetric=False`
        * `symmetric=False` is not compatible with `per_channel=True`
        """
        if not symmetric and self.per_channel:
            log.warning(
                "asymmetric per channel quantization is not supported.",
                stacklevel=2,
            )
            return False
        if symmetric and self.learn_zero_point:
            log.warning(
                "`learn_zero_point` will be automatically set to False as `symmetric` is being set to True",
                stacklevel=2,
            )
            self.learn_zero_point = False
        if not symmetric and self.ptq_calibration == PTQCalibrationType.absmax:
            log.warning(
                "`ptq_calibration` will be automatically set to `minmax` as `symmetric` is being set to False",
                stacklevel=2,
            )
            self.ptq_calibration = PTQCalibrationType.minmax
        return True

    def check_ptq_calibration(self, ptq_calibration: PTQCalibrationType) -> bool:
        """
        * if `symmetric=False`, `ptq_calibration` must not be 'absmax'
        * if `ptq_calibration="percentile"` and `percentile` is None, it will be automatically set to 99.99
        """
        if not self.symmetric and ptq_calibration == PTQCalibrationType.absmax:
            return False
        if ptq_calibration == PTQCalibrationType.percentile and self.percentile is None:
            log.warning(
                '`ptq_calibration="percentile"` requires a `percentile` value.'
                "Will set `percentile` to 99.99 automatically.",
                stacklevel=2,
            )
            with log.ignore_warnings():
                self.percentile = 99.99
        if ptq_calibration != PTQCalibrationType.percentile and self.percentile is not None:
            log.warning(
                '`percentile` is used only when `ptq_calibration="percentile"`.'
                f"The percentile value {self.percentile} will be ignored.",
                stacklevel=2,
            )
        return True

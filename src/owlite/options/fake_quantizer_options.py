from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, PlainSerializer, model_validator
from typing_extensions import Self

from ..enums import PTQCalibrationType, QATBackwardType, get_before_validator, serialize_as_name
from ..owlite_core.logger import log
from .options_mixin import OptionsMixin

Percentile = Annotated[float, Field(gt=0, le=100)]
Precision = Literal[4, 8, 16]
PTQCalibration = Annotated[
    PTQCalibrationType,
    Field(default=PTQCalibrationType.absmax),
    BeforeValidator(get_before_validator(PTQCalibrationType)),
    PlainSerializer(serialize_as_name),
]
QATBackward = Annotated[
    QATBackwardType,
    Field(default=QATBackwardType.ste),
    BeforeValidator(get_before_validator(QATBackwardType)),
    PlainSerializer(serialize_as_name),
]


# pylint: disable=too-many-instance-attributes
class FakeQuantizerOptions(OptionsMixin):
    """Options required for setting up a quantizer."""

    qat_backward: QATBackward
    ptq_calibration: PTQCalibration
    precision: Precision = Field(default=8)
    per_channel: bool = Field(default=False)
    symmetric: bool = Field(default=True)
    learn_zero_point: bool = Field(default=False)
    unsigned: bool = Field(default=False)
    grad_scale: float = Field(default=1.000, ge=0, le=1)
    percentile: Percentile | None = Field(default=None)

    @classmethod
    def ste_per_channel(cls, **kwargs: Any) -> Self:
        """Create per-channel options with "ste" backward."""
        return cls(qat_backward=QATBackwardType.ste, per_channel=True, **kwargs)

    @classmethod
    def ste_per_tensor(cls, **kwargs: Any) -> Self:
        """Create per-tensor options with "ste" backward."""
        return cls(qat_backward=QATBackwardType.ste, per_channel=False, **kwargs)

    @classmethod
    def clq_per_channel(cls, **kwargs: Any) -> Self:
        """Create per-channel options with "clq" backward."""
        return cls(qat_backward=QATBackwardType.clq, per_channel=True, **kwargs)

    @classmethod
    def clq_per_tensor(cls, **kwargs: Any) -> Self:
        """Create per-tensor options with "clq" backward."""
        return cls(qat_backward=QATBackwardType.clq, per_channel=False, **kwargs)

    @model_validator(mode="after")
    def validate_attribute_dependencies(self) -> "FakeQuantizerOptions":
        """Validate the dependencies between the attributes."""
        # symmetric sits on the highest priority
        if self.symmetric:
            if self.learn_zero_point:
                log.warning(
                    "learn_zero_point will be automatically set to False as symmetric is True. "
                    "If you want to set learn_zero_point to True, set symmetric to False in advance."
                )
                self.learn_zero_point = False
        else:
            if self.ptq_calibration == PTQCalibrationType.absmax:
                log.warning(
                    "ptq_calibration will be automatically set to minmax as symmetric is False. "
                    'If you want to set ptq_calibration to "absmax", then set symmetric to True in advance'
                )
                self.ptq_calibration = PTQCalibrationType.minmax
            if self.per_channel:
                log.warning(
                    "per_channel will be automatically set to False as symmetric is False. "
                    "If you want to set per_channel to True, then set symmetric to True in advance"
                )
                self.per_channel = False

        if self.ptq_calibration == PTQCalibrationType.percentile and self.percentile is None:
            log.warning('percentile is required when ptq_calibration="percentile". Will set it to 99.9')
            self.percentile = 99.9

        if self.ptq_calibration != PTQCalibrationType.percentile and self.percentile is not None:
            log.warning(
                'percentile is used only when ptq_calibration="percentile". '
                f"The given percentile value {self.percentile} will be ignored and automatically set to None instead",
            )
            self.percentile = None

        return self

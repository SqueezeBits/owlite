from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, PlainSerializer, model_validator
from typing_extensions import Self

from ..core.logger import log
from ..enums import PTQCalibrationType, QATBackwardType, TargetDType, get_before_validator, serialize_as_name
from .options_mixin import OptionsMixin

Dtype = Annotated[
    TargetDType,
    Field(default=TargetDType.int8),
    BeforeValidator(get_before_validator(TargetDType)),
    PlainSerializer(serialize_as_name),
]
Percentile = Annotated[float, Field(gt=0, le=100)]
Precision = Literal[8, 16]
PTQCalibration = Annotated[
    PTQCalibrationType,
    Field(default=PTQCalibrationType.minmax),
    BeforeValidator(get_before_validator(PTQCalibrationType)),
    PlainSerializer(serialize_as_name),
]
QATBackward = Annotated[
    QATBackwardType,
    Field(default=QATBackwardType.ste),
    BeforeValidator(get_before_validator(QATBackwardType)),
    PlainSerializer(serialize_as_name),
]


def map_precision_and_unsigned_to_dtype(precision: Precision, unsigned: bool) -> Dtype:
    """Map the combination of precision and unsigned flag to dtype.

    Args:
        precision (Precision): The precision.
        unsigned (bool): The unsigned flag.

    Returns:
        Dtype: The mapped data type.
    """
    if precision == 8 and unsigned:
        return TargetDType.uint8
    if precision == 8 and not unsigned:
        return TargetDType.int8
    return TargetDType.fp16


# pylint: disable=too-many-instance-attributes
class FakeQuantizerOptions(OptionsMixin):
    """Options required for setting up a quantizer."""

    dtype: Dtype
    qat_backward: QATBackward
    ptq_calibration: PTQCalibration
    per_channel: bool = Field(default=False)
    symmetric: bool = Field(default=True)
    learn_zero_point: bool = Field(default=False)
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

    @model_validator(mode="before")
    @classmethod
    def handle_deprecated_fields(cls, data: Any) -> Any:
        """Handle deprecated fields.

        deprecated fields since version 1.2.0
        * precision: Precision = Field(default=8)
        * unsigned: bool = Field(default=False)
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected a dictionary data but got {data}")
        if ("precision" in data) ^ ("unsigned" in data):
            raise ValueError("'unsigned' and 'precision' field should always be used together")
        if "dtype" in data and ("precision" in data or "unsigned" in data):
            raise ValueError("'dtype' field cannot be used together with 'precision' or 'unsigned' field")
        if not ("precision" in data and "unsigned" in data):
            return data
        if not (
            (precision := data.pop("precision", None)) in Precision.__args__  # type: ignore[attr-defined]
            and isinstance((unsigned := data.pop("unsigned", None)), bool)
        ):
            raise ValueError(
                f"Invalid values found in 'precision' and/or 'unsigned' field: precision={precision}, "
                f"unsigned={unsigned}"
            )
        data["dtype"] = map_precision_and_unsigned_to_dtype(precision, unsigned)
        log.debug_warning(
            f"An older version of config was detected: "
            f"precision({precision}), unsigned({unsigned}) -> {data['dtype'].name}"
        )
        return data

    @model_validator(mode="after")
    def validate_attribute_dependencies(self) -> Self:
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

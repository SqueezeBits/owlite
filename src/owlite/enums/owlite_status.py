from enum import Enum


class OwLiteStatus(Enum):
    """The enum for specifying model status about compression with `GraphModule.meta`

    Attributes
        NOT_COMPRESSED: The model is symbolic traced, but before inserting quantizers.
        COMPRESSED: The quantizers are inserted into the model, but not calibrated.
        CALIBRATED: The calibration of the quantizers in the model is completed.
    """

    NOT_COMPRESSED = 0
    COMPRESSED = 1
    CALIBRATED = 2

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

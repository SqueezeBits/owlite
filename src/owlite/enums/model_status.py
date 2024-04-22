from enum import IntEnum


class ModelStatus(IntEnum):
    """The enum for specifying model status about compression with `GraphModule.meta`.

    Attributes:
        TRACED: The model is traced, but not compressed.
        COMPRESSED: The model is compressed, but not calibrated.
        CALIBRATED: The model is calibrated with `owlite.calibrate`.
    """

    TRACED = 0
    COMPRESSED = 1
    CALIBRATED = 2

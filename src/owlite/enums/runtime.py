from enum import IntEnum


# pylint: disable=invalid-name
class Runtime(IntEnum):
    """Runtimes supported by OwLite."""

    Unknown = 0
    TensorRT = 1
    FuriosaSDK = 2

    @property
    def simulate_int32_bias(self) -> bool:
        """Whether or not this runtime requires int32 bias simulation."""
        return self in (Runtime.FuriosaSDK,)

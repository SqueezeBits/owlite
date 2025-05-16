from enum import IntEnum


# pylint: disable=invalid-name
class Runtime(IntEnum):
    """Runtimes supported by OwLite."""

    Unknown = 0
    TensorRT = 1
    FuriosaSDK = 2
    RebelionSDK = 3
    QNN = 4

    @property
    def simulate_int32_bias(self) -> bool:
        """Whether or not this runtime requires int32 bias simulation."""
        return self in (Runtime.FuriosaSDK, Runtime.QNN)

    @property
    def file_ext(self) -> str:
        """File extension of the runtime binary."""
        match self.value:
            case Runtime.TensorRT:
                return "engine"

            case Runtime.QNN:
                return "qnn.bin"

        return "engine"

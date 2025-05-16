from pydantic import BaseModel, Field
from typing_extensions import Self

from ..core.cache.device import Device
from ..core.logger import log
from ..enums import Runtime


class ONNXExportOptions(BaseModel):
    """Class handling options for ONNX export.

    OwLite internally imports the target model to ONNX during conversion or benchmarking.
    Users can set options for ONNX export using this class.
    """

    opset_version: int = Field(default=17)

    @classmethod
    def create(cls, device: Device) -> Self:
        """Create a ONNXExportOptions for export on `device`.

        Args:
            device (Device): The device to benchmark.

        Returns:
            Self: Additional options for exporting ONNX compatibility with the device.
        """
        match device.runtime:
            case Runtime.TensorRT:
                return cls()

            case Runtime.FuriosaSDK:
                log.info(
                    "ONNX opset version will be automatically set to 13 for the compatibility with Furiosa SDK"
                )  # UX
                return cls(opset_version=13)

            case Runtime.QNN:
                return cls()

            case _:
                log.warning("Unknown device, using default ONNX export options")  # UX
                return cls()

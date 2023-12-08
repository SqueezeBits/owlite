from dataclasses import dataclass


@dataclass
class ONNXExportOptions:
    """
    Class handling options for ONNX export.

    OwLite internally imports the target model to ONNX during conversion or benchmarking.
    Users can set options for ONNX export using this class.
    """

    opset_version: int = 17

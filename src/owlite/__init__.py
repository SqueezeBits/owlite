import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

from .core.constants import OWLITE_VERSION as __version__  # noqa: N811

_import_structure = {
    "backend": [
        "fx",
        "onnx",
    ],
    "calibrators": [
        "CalibrationContext",
        "calibrate",
    ],
    "compression": ["compress"],
    "enums": [
        "PTQCalibrationType",
        "QATBackwardType",
        "Runtime",
    ],
    "nn": [
        "FakePerChannelINTQuantizer",
        "FakePerTensorINTQuantizer",
        "FakePerChannelFPQuantizer",
        "FakePerTensorFPQuantizer",
        "FakeINTQuantizer",
        "FakeQuantizer",
        "QConv1d",
        "QConv2d",
        "QConv3d",
        "QLinear",
        "QConvBn1d",
        "QConvBn2d",
        "QConvBn3d",
    ],
    "options": [
        "Channel",
        "CompressionOptions",
        "DynamicAxisOptions",
        "DynamicInputOptions",
        "FakeQuantizerOptions",
        "GraphQuantizationOptions",
        "NodeQuantizationOptions",
        "ONNXExportOptions",
    ],
    "owlite": ["init"],
}

if TYPE_CHECKING:
    from .backend import fx, onnx
    from .calibrators import (
        CalibrationContext,
        calibrate,
    )
    from .compression import compress
    from .enums import PTQCalibrationType, QATBackwardType, Runtime
    from .nn import (
        FakePerChannelFPQuantizer,
        FakePerChannelINTQuantizer,
        FakePerTensorFPQuantizer,
        FakePerTensorINTQuantizer,
        FakeQuantizer,
        QConv1d,
        QConv2d,
        QConv3d,
        QConvBn1d,
        QConvBn2d,
        QConvBn3d,
        QLinear,
    )
    from .options import (
        Channel,
        CompressionOptions,
        DynamicAxisOptions,
        DynamicInputOptions,
        FakeQuantizerOptions,
        GraphQuantizationOptions,
        NodeQuantizationOptions,
        ONNXExportOptions,
    )
    from .owlite import init
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects={"__version__": __version__},
    )

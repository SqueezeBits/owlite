"""owlite

Easily quantize pytorch models and apply various QAT and PTQ methods.
The quantized model is then exported to onnx for conversion to TRT.
"""
from . import api, calib, nn
from .backend import fx, onnx
from .calibrators import (
    CalibrationContext,
    calibrate,
    prepare_for_calibration,
    update_fake_quantizers,
)
from .enums import PTQCalibrationType, QATBackwardType
from .logger import log
from .nn.fake_quantizer import enable_quantizers
from .options import (
    FakeQuantizerOptions,
    GraphQuantizationOptions,
    NodeQuantizationOptions,
    ONNXExportOptions,
)
from .owlite import init
from .quantize import quantize

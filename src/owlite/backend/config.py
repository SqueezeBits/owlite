import os

# Flag to disable automatic object monkey patching
DISABLE_AUTO_PATCH = os.environ.get("OWLITE_DISABLE_AUTO_PATCH", "0") == "1"

# Maximum iteration limit for ONNX transformations.
FX_TRANSFORM_MAXIMUM_ITERATION = int(os.environ.get("OWLITE_FX_TRANSFORM_MAXIMUM_ITERATION", 100))

# Maximum iteration limit for ONNX transformations.
ONNX_TRANSFORM_MAXIMUM_ITERATION = int(os.environ.get("OWLITE_ONNX_TRANSFORM_MAXIMUM_ITERATION", 100))

# Run strict shape inference
STRICT_ONNX_SHAPE_INFERENCE = os.environ.get("OWLITE_STRICT_ONNX_SHAPE_INFERENCE", "1") == "1"

# Run strict invariance checking
STRICT_ONNX_FUNCTIONALITY_CHECKING = os.environ.get("OWLITE_STRICT_ONNX_FUNCTIONALITY_CHECKING", "1") == "1"



# [DISCLAIMER] Configurations below are deprecated and may be removed in later versions.

# (deprecated since 2.2.0) ONNX operator types to save input parameters internally during onnx export.
# List entry can be either
#   1) a operator type in string
#   2) a tuple of operator type in string and tuple of indices of inputs to store internally
#
# When an index tuple is provided, the input parameters not included in the tuple will be stored externally.
ONNX_OPS_TO_SAVE_PARAMETERS_INTERNALLY: list[tuple[str, list[int]] | str] = []

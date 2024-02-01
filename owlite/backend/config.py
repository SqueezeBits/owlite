import os

# This flag ensures module output(return value of forward) consistency between before and after trace.
FORCE_OUTPUT_COMPATIBILITY = os.environ.get("OWLITE_FORCE_OUTPUT_COMPATIBILITY", "1") == "1"

# Maximum iteration limit for ONNX transformations.
ONNX_TRANSFORM_MAXIMUM_ITERATION = int(os.environ.get("OWLITE_ONNX_TRANSFORM_MAXIMUM_ITERATION", 100))

# ONNX operator types to save parameters internally during onnx export.
ONNX_OPS_TO_SAVE_PARAMETERS_INTERNALLY = (
    "Col2Im",
    "Compress",
    "ConstantOfShape",
    "CumSum",
    "Expand",
    "Gather",
    "GatherElements",
    "GatherND",
    "GridSample",
    "Pad",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
    "Reshape",
    "Resize",
    "Scatter",
    "ScatterElements",
    "ScatterND",
    "Shape",
    "Slice",
    "Split",
    "Squeeze",
    "Tile",
    "TopK",
    "Unsqueeze",
)

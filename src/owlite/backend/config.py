import os

# Flag to disable automatic object monkey patching
DISABLE_AUTO_PATCH = os.environ.get("OWLITE_DISABLE_AUTO_PATCH", "0") == "1"

# Flag to enforce graph module consistency between the module before trace and the graph module after trace.
# Turning this flag off should only be done for debugging purpose as OwLite model checking logic will fail without this.
FORCE_GRAPH_MODULE_COMPATIBILITY = os.environ.get("OWLITE_FORCE_GRAPH_MODULE_COMPATIBILITY", "1") == "1"

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

# pylint: disable=c-extension-no-member, broad-exception-caught

import onnx
from onnx import ModelProto

from ... import capi  # type: ignore[attr-defined]
from ...core.logger import log
from ..config import ONNX_TRANSFORM_MAXIMUM_ITERATION


def optimize(
    model_proto: ModelProto,
    *,
    max_num_iters: int = ONNX_TRANSFORM_MAXIMUM_ITERATION,
    skipped_optimizers: list[str] | None = None,
) -> ModelProto:
    """Apply graph-level optimization to the computation graph encapsulated by the given ONNX model proto.

    Args:
        model_proto (ModelProto): a model proto to optimize.
        max_num_iters (int, optional): the maximum number of iterations to apply the set of optimization passes.
            Defaults to ONNX_TRANSFORM_MAXIMUM_ITERATION, which can be set via the environment variable
            `OWLITE_ONNX_TRANSFORM_MAXIMUM_ITERATION`.
        skipped_optimizers (list[str] | None, optional): the names of optimization passes to skip.
            Defaults to None.

    Returns:
        ModelProto: the ONNX model proto containing the optimized graph.
    """
    try:
        model_proto_bytes = capi.optimize(
            model_proto.SerializeToString(),
            skipped_optimizers,
            max_num_iters,
        )
        return onnx.load_from_string(model_proto_bytes)
    except Exception as e:
        log.warning(f"Failed to optimize ONNX: {e}")
        return model_proto


def optimize_path(
    input_path: str,
    output_path: str,
    *,
    max_num_iters: int = ONNX_TRANSFORM_MAXIMUM_ITERATION,
    skipped_optimizers: list[str] | None = None,
) -> str:
    """Apply graph-level optimization to the computation graph encapsulated by the given ONNX model proto.

    Same as `owlite.onnx.optimize` but involves file I/O. (Required for models larger than 2GB.)

    Args:
        input_path (str): the path to the input model proto file.
        output_path (str): the path to the output model proto file to be created.
        max_num_iters (int, optional): the maximum number of iterations to apply the set of optimization passes.
            Defaults to ONNX_TRANSFORM_MAXIMUM_ITERATION, which can be set via the environment variable
            `OWLITE_ONNX_TRANSFORM_MAXIMUM_ITERATION`.
        skipped_optimizers (list[str] | None, optional): the names of optimization passes to skip.
            Defaults to None.

    Returns:
        str: `input_path` if optimization fails, `output_path` otherwise.
    """
    if input_path == output_path:
        log.error("You must provide different input_path and output_path to `owlite.onnx.optimize_path`")  # UX
        raise ValueError("Inplace ONNX optimization via file is not supported.")  # UX
    try:
        capi.optimize_path(input_path, output_path, skipped_optimizers, max_num_iters)
        return output_path
    except Exception as e:
        log.warning(f"Failed to optimize ONNX: {e}")
        return input_path

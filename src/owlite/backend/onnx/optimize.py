# pylint: disable=c-extension-no-member, broad-exception-caught

import os

import onnx
from onnx import ModelProto

from ... import capi  # type: ignore[attr-defined]
from ...core.logger import log
from ..config import (
    ONNX_EXTERNAL_DATA_SIZE_THRESHOLD,
    ONNX_TRANSFORM_MAXIMUM_ITERATION,
)


def optimize(
    model_proto: ModelProto,
    *,
    max_num_iters: int = ONNX_TRANSFORM_MAXIMUM_ITERATION,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    skipped_optimizers: list[str] | None = None,
) -> ModelProto:
    """Apply graph-level optimization to the computation graph encapsulated by the given ONNX model proto.

    Args:
        model_proto (ModelProto): a model proto to optimize.
        max_num_iters (int, optional): the maximum number of iterations to apply the set of optimization passes.
            Defaults to ONNX_TRANSFORM_MAXIMUM_ITERATION, which can be set via the environment variable
            `OWLITE_ONNX_TRANSFORM_MAXIMUM_ITERATION`.
        input_names (list[str] | None, optional): the names of input tensors.
            Defaults to None.
        output_names (list[str] | None, optional): the names of output tensors.
            Defaults to None.
        skipped_optimizers (list[str] | None, optional): the names of optimization passes to skip.
            Defaults to None.

    Returns:
        ModelProto: the ONNX model proto containing the optimized graph.
    """
    try:
        model_proto_bytes = capi.optimize(
            model_proto.SerializeToString(),
            input_names,
            output_names,
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
    size_threshold: int = ONNX_EXTERNAL_DATA_SIZE_THRESHOLD,
    max_num_iters: int = ONNX_TRANSFORM_MAXIMUM_ITERATION,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    skipped_optimizers: list[str] | None = None,
) -> str:
    """Apply graph-level optimization to the computation graph encapsulated by the given ONNX model proto.

    Same as `owlite.onnx.optimize` but involves file I/O. (Required for models larger than 2GB.)

    Args:
        input_path (str): the path to the input model proto file.
        output_path (str): the path to the output model proto file to be created.
        size_threshold (int, optional): the lower bound in bytes for the large tensors to save externally.
            Defaults to ONNX_EXTERNAL_DATA_SIZE_THRESHOLD, which can be set via the environment variable
            `OWLITE_ONNX_EXTERNAL_DATA_SIZE_THRESHOLD`.
        max_num_iters (int, optional): the maximum number of iterations to apply the set of optimization passes.
            Defaults to ONNX_TRANSFORM_MAXIMUM_ITERATION, which can be set via the environment variable
            `OWLITE_ONNX_TRANSFORM_MAXIMUM_ITERATION`.
        input_names (list[str] | None, optional): the names of input tensors.
            Defaults to None.
        output_names (list[str] | None, optional): the names of output tensors.
            Defaults to None.
        skipped_optimizers (list[str] | None, optional): the names of optimization passes to skip.
            Defaults to None.

    Returns:
        str: `input_path` if optimization fails, `output_path` otherwise.
    """
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    if input_path == output_path:
        log.error("You must provide different input_path and output_path to `owlite.onnx.optimize_path`")  # UX
        raise ValueError("Inplace ONNX optimization via file is not supported.")  # UX
    try:
        output_prefix, _ = os.path.splitext(output_path)
        if os.path.isfile(external_data_path := f"{output_prefix}.bin"):
            log.warning(f"External data file at {external_data_path} will be overwritten.")  # UX
            os.remove(external_data_path)
        location = os.path.basename(external_data_path)
        capi.optimize_path(
            input_path,
            output_path,
            location,
            size_threshold,
            input_names,
            output_names,
            skipped_optimizers,
            max_num_iters,
        )
        return output_path
    except Exception as e:
        log.warning(f"Failed to optimize ONNX: {e}")
        return input_path

import os
import sys
import uuid
from collections.abc import Iterable
from itertools import chain
from typing import Any

import onnx
from onnx import GraphProto, ModelProto, TensorProto
from onnx.external_data_helper import (
    _get_attribute_tensors_from_graph,
    _get_initializer_tensors_from_graph,
    _is_valid_filename,
    save_external_data,
    set_external_data,
    uses_external_data,
)

from ..config import ONNX_EXTERNAL_DATA_SIZE_THRESHOLD


def _get_all_tensors_from_graph(graph: GraphProto) -> Iterable[TensorProto]:
    return chain(
        _get_initializer_tensors_from_graph(graph),
        _get_attribute_tensors_from_graph(graph),
    )


def export_with_external_data(
    model_proto: ModelProto,
    output_path: str,
    *,
    all_tensors_to_one_file: bool = True,
    location: str | None = None,
    size_threshold: int = ONNX_EXTERNAL_DATA_SIZE_THRESHOLD,
    convert_attribute: bool = False,
    ops_to_save_parameter_internally: list[tuple[str, list[int]] | str] | None = None,
    **kwargs: Any,
) -> None:
    """Export an onnx-graphsurgeon Graph or the ModelProto to an ONNX model with external data.

    Args:
        model_proto (ModelProto): The ONNX graph or model proto to export
        output_path (str): The path to save ONNX file at.
            (The base directory will be automatically created if it doesn't exist.)
        all_tensors_to_one_file (bool, optional): If true, save all tensors to one external file specified by location.
            If false, save each tensor to a file named with the tensor name. Defaults to True.
        location (str | None, optional): Specify the external file that all tensors to save to.
            If not specified, will use the model name. Defaults to None.
        size_threshold (int, optional): Threshold for size of data. Only when tensor's data is >= the size_threshold
            it will be converted to external data. To convert every tensor with raw data to external data set
            size_threshold=0. Defaults to ONNX_EXTERNAL_DATA_SIZE_THRESHOLD, which can be set via the environment
            variable `OWLITE_ONNX_EXTERNAL_DATA_SIZE_THRESHOLD`.
        convert_attribute (bool, optional): If true, convert all tensors to external data.
            If false, convert only non-attribute tensors to external data. Defaults to False.
        ops_to_save_parameter_internally (list[tuple[str, list[int]] | str] | None, optional): (deprecated) ONNX
            operation types to store parameter within the ONNX file. Defaults to None.
        kwargs: Additional arguments to onnx.helper.make_model.

    Returns:
        ModelProto: A corresponding ONNX model.
    """
    graph_proto = model_proto.graph
    base_path = os.path.dirname(output_path)
    if location is None:
        base_name = os.path.basename(output_path)
        location = f"{os.path.splitext(base_name)[0]}.bin"
    if base_path:
        os.makedirs(base_path, exist_ok=True)

    convert_graph_to_external_data(
        graph_proto,
        base_path=base_path,
        all_tensors_to_one_file=all_tensors_to_one_file,
        size_threshold=size_threshold,
        convert_attribute=convert_attribute,
        location=location,
        ops_to_save_parameter_internally=ops_to_save_parameter_internally,
    )

    if "opset_imports" not in kwargs:
        kwargs["opset_imports"] = model_proto.opset_import

    exported_model_proto = onnx.helper.make_model(graph_proto, **kwargs)
    exported_model_proto.producer_name = model_proto.producer_name
    exported_model_proto.producer_version = model_proto.producer_version

    onnx.save(exported_model_proto, output_path)


def convert_graph_to_external_data(
    graph: GraphProto,
    base_path: str,
    *,
    all_tensors_to_one_file: bool = True,
    location: str | None = None,
    size_threshold: int = ONNX_EXTERNAL_DATA_SIZE_THRESHOLD,
    convert_attribute: bool = False,
    ops_to_save_parameter_internally: list[tuple[str, list[int]] | str] | None = None,
) -> None:
    """Set all tensors with raw data as external data.

    To save tensor data externally, this function should be called before save_model
    as `onnx.save_model` will save all the tensor data as external data after calling this function.

    Args:
        graph (GraphProto): Graph to be converted.
        base_path (str): System path of a folder where tensor data is to be stored.
        all_tensors_to_one_file (bool, optional): If true, save all tensors to one external file specified by location.
            If false, save each tensor to a file named with the tensor name. Defaults to True.
        location (str | None, optional): Specify the external file that all tensors to save to.
            If not specified, will use the model name. Defaults to None.
        size_threshold (int, optional): Threshold for size of data. Only when tensor's data is >= the size_threshold
            it will be converted to external data. To convert every tensor with raw data to external data
            set size_threshold=0. Defaults to ONNX_EXTERNAL_DATA_SIZE_THRESHOLD, which can be set via the environment
            variable `OWLITE_ONNX_EXTERNAL_DATA_SIZE_THRESHOLD`.
        convert_attribute (bool, optional): If true, convert all tensors to external data.
            If false, convert only non-attribute tensors to external data. Defaults to False.
        ops_to_save_parameter_internally (list[tuple[str, list[int]] | str] | None, optional): (deprecated) ONNX
            operation types to store parameter within the ONNX file. Defaults to None.
    """
    tensors = _get_all_tensors_from_graph(graph) if convert_attribute else _get_initializer_tensors_from_graph(graph)

    tensor_names_to_save_internally: list[str] = []
    if ops_to_save_parameter_internally:
        tensor_export_configuration = {
            entry if (is_str := isinstance(entry, str)) else entry[0]: [] if is_str else entry[1]
            for entry in ops_to_save_parameter_internally
        }
        for node in graph.node:
            if (config := tensor_export_configuration.get(node.op_type, None)) is None:
                continue
            tensor_names_to_save_internally.extend([node.input[i] for i in config] if len(config) > 0 else node.input)

    if all_tensors_to_one_file:
        file_name = str(uuid.uuid1())
        if location:
            file_name = location
        for tensor in tensors:
            if (
                tensor.name not in tensor_names_to_save_internally
                and tensor.HasField("raw_data")
                and sys.getsizeof(tensor.raw_data) >= size_threshold
            ):
                set_external_data(tensor, file_name)
    else:
        for tensor in tensors:
            if (
                tensor.name not in tensor_names_to_save_internally
                and tensor.HasField("raw_data")
                and sys.getsizeof(tensor.raw_data) >= size_threshold
            ):
                tensor_location = tensor.name
                if not _is_valid_filename(tensor_location):
                    tensor_location = str(uuid.uuid1())
                set_external_data(tensor, tensor_location)

    for tensor in _get_all_tensors_from_graph(graph):
        # Writing to external data happens in 2 passes:
        # 1. Tensors with raw data which pass the necessary conditions (size threshold etc) are marked for serialization
        # 2. The raw data in these tensors is serialized to a file
        # Thus serialize only if tensor has raw data and it was marked for serialization
        if uses_external_data(tensor) and tensor.HasField("raw_data"):
            save_external_data(tensor, base_path=base_path)
            tensor.ClearField("raw_data")

import os
import sys
import uuid
from collections.abc import Iterable
from functools import reduce
from itertools import chain
from typing import Any

import onnx
import onnx_graphsurgeon as gs
from onnx import GraphProto, ModelProto, TensorProto
from onnx.external_data_helper import (
    _get_attribute_tensors_from_graph,
    _get_initializer_tensors_from_graph,
    _is_valid_filename,
    save_external_data,
    set_external_data,
    uses_external_data,
)
from onnx_graphsurgeon.exporters.onnx_exporter import OnnxExporter

from ..config import ONNX_OPS_TO_SAVE_PARAMETERS_INTERNALLY


def _get_all_tensors_from_graph(graph: GraphProto) -> Iterable[TensorProto]:
    return chain(
        _get_initializer_tensors_from_graph(graph),
        _get_attribute_tensors_from_graph(graph),
    )


def export_with_external_data(
    graph_or_model_proto: gs.Graph | ModelProto,
    output_path: str,
    do_type_check: bool = True,
    all_tensors_to_one_file: bool = True,
    location: str | None = None,
    size_threshold: int = 1024,
    convert_attribute: bool = False,
    **kwargs: Any,
) -> None:
    """Export an onnx-graphsurgeon Graph to an ONNX model with external data.

    Args:
        graph_or_model_proto (ModelProto | gs.Graph): The ONNX graph or model proto to export
        output_path (str): The path to save ONNX file at.
            (The base directory will be automatically created if it doesn't exist.)
        do_type_check (bool): Whether to check that input and output tensors have data types defined, and fail if not.
        all_tensors_to_one_file (bool, optional): If true, save all tensors to one external file specified by location.
            If false, save each tensor to a file named with the tensor name. Defaults to True.
        location (str | None, optional): Specify the external file that all tensors to save to.
            If not specified, will use the model name. Defaults to None.
        size_threshold (int, optional): Threshold for size of data. Only when tensor's data is >= the size_threshold
            it will be converted to external data. To convert every tensor with raw data to external data
            set size_threshold=0. Defaults to 1024.
        convert_attribute (bool, optional): If true, convert all tensors to external data.
            If false, convert only non-attribute tensors to external data. Defaults to False.
        kwargs: Additional arguments to onnx.helper.make_model.

    Returns:
        ModelProto: A corresponding ONNX model.
    """
    is_gs_graph = isinstance(graph_or_model_proto, gs.Graph)

    graph_proto = (
        OnnxExporter.export_graph(graph_or_model_proto, do_type_check=do_type_check)
        if is_gs_graph
        else graph_or_model_proto.graph
    )
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
    )

    if "opset_imports" not in kwargs:
        if is_gs_graph and graph_or_model_proto.import_domains is None:
            kwargs["opset_imports"] = [onnx.helper.make_opsetid("", graph_or_model_proto.opset)]
        else:
            kwargs["opset_imports"] = (
                graph_or_model_proto.import_domains if is_gs_graph else graph_or_model_proto.opset_import
            )

    model = onnx.helper.make_model(graph_proto, **kwargs)
    model.producer_name = graph_or_model_proto.producer_name
    model.producer_version = graph_or_model_proto.producer_version

    onnx.save(
        model,
        output_path,
    )


def convert_graph_to_external_data(
    graph: GraphProto,
    base_path: str,
    all_tensors_to_one_file: bool = True,
    location: str | None = None,
    size_threshold: int = 1024,
    convert_attribute: bool = False,
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
            set size_threshold=0. Defaults to 1024.
        convert_attribute (bool, optional): If true, convert all tensors to external data.
            If false, convert only non-attribute tensors to external data. Defaults to False.
    """
    tensors = _get_all_tensors_from_graph(graph) if convert_attribute else _get_initializer_tensors_from_graph(graph)

    tensor_names_to_save_internally: list[str] = reduce(
        lambda acc, cur: acc + list(cur.input) if cur.op_type in ONNX_OPS_TO_SAVE_PARAMETERS_INTERNALLY else acc,
        graph.node,
        [],
    )
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

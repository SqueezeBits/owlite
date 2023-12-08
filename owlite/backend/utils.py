"""Utilities for tracing torch.fx.Graph and for exporting torch.nn.Module into ONNX"""
# ruff: noqa: E741
import inspect
import json
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable
from functools import reduce
from numbers import Number
from typing import Any, Optional, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from onnx import ModelProto, TensorProto, TypeProto, ValueInfoProto
from onnx import NodeProto as ONNXNode
from onnx_graphsurgeon.importers.onnx_importer import get_numpy_type
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node as FXNode
from torch.fx.node import Target as FXTarget

from ..logger import log

AnyNode = Union[FXNode, ONNXNode, gs.Node]
ONNXInputSignature = list[tuple[str, list[Union[str, int]]]]

RTOL_FP16 = np.finfo(np.float16).smallest_normal.item()
RTOL_FP32 = 1.0e-5
ATOL_FP16 = np.finfo(np.float16).eps.item()
ATOL_FP32 = 1.0e-8


# pylint:disable = invalid-name
def nodestr(node: Optional[AnyNode], show_activations: bool = False) -> str:
    """Generates the string representation of a node instance

    Args:
        node (Optional[AnyNode]): a node. Must be an instance of one of the types:
            torch.fx.Node, onnx.NodeProto or gs.Node
        show_activations (bool, optional): Only available if node is either onnx.NodeProto or gs.Node instance. If True,
            the string representation contains the information about the node's input and output activations.
            Defaults to False.

    Returns:
        str: the string representation of the node
    """
    if node is None:
        return "<node-not-found>"
    if isinstance(node, ONNXNode):
        s = f"{node.name} ({node.op_type})"
        if show_activations:
            a = json.dumps(
                {"inputs": list(node.input), "outputs": list(node.output)},
                indent=2,
                sort_keys=True,
            )
            s = f"{s}: {a}"
        return s
    if isinstance(node, FXNode):
        if (
            node.op == "call_module"
            and isinstance(node.target, str)
            and isinstance(node.graph.owning_module, GraphModule)
        ):
            target = node.graph.owning_module.get_submodule(node.target)
        else:
            target = node.target
        s = f"{node.name}: {node.op}({targetstr(target)})"
        if show_activations:
            a = json.dumps(
                {
                    "args": f"{node.args}",
                    "kwargs": f"{node.kwargs}",
                    "inputs": [*map(nodestr, node.all_input_nodes)],
                    "outputs": [*map(nodestr, node.users)],
                },
                indent=2,
                sort_keys=True,
            )
            s = f"{s}: {a}"
        return s
    if isinstance(node, gs.Node):
        s = f"{node.name} ({node.op})"
        if show_activations:
            a = json.dumps(
                {
                    "inputs": [*(t.name for t in node.inputs)],
                    "outputs": [*(t.name for t in node.outputs)],
                },
                indent=2,
                sort_keys=True,
            )
            s = f"{s}: {a}"
        return s
    return "<not-a-node>"


def targetstr(target: FXTarget) -> str:
    """Generates the string representation of the target of a torch.fx.Node

    Args:
        target (FXTarget): the target of a torch.fx.Node instance.

    Returns:
        str: the string representation of the target
    """
    if hasattr(target, "__module__") and hasattr(target, "__name__"):
        return f"{target.__module__}.{target.__name__}"
    if isinstance(target, str) and hasattr(torch.Tensor, target):
        return f"torch.Tensor.{target}"
    return f"{target}"


def typestr(tensor: torch.Tensor) -> str:
    """Generates the MLIR-like string representation of the type of a torch.Tensor instance.

    Args:
        tensor (torch.Tensor): a tensor

    Returns:
        str: the string representation of the type of the tensor
    """
    return f'{"x".join(map(str, tensor.shape))} ({str(tensor.dtype).removeprefix("torch.")})'


def camel_to_snake(camel_cased_string: str) -> str:
    """Converts given camelCase string to snake_case string

    Args:
        camel_cased_string (str): string to convert

    Returns:
        str: converted string in snake_case
    """
    snake_cased_string = ""
    previous_length: int = 0
    previous_char = ""
    for i, c in enumerate(camel_cased_string):
        if not c.isupper():
            snake_cased_string += c
            previous_length += 1
            previous_char = c
            continue
        if i > 0 and previous_length > 0 and previous_char.isalpha():
            snake_cased_string += "_"
            previous_length = 0
            previous_char = c
        snake_cased_string += c.lower()
    return snake_cased_string


def get_most_common_device(model: torch.nn.Module) -> torch.device:
    """Finds the most common device where the parameters of the model reside.

    Args:
        model (torch.nn.Module): a model

    Returns:
        torch.device: the most common device where the parameters of the model reside.
    """
    counter = Counter(p.device for p in model.parameters())
    if len(counter) == 0:
        return torch.device("cpu")
    if len(counter) > 1:
        log.warning(f"The model parameters reside on more than 1 devices: {set(counter.elements())}")
    return counter.most_common(1)[0][0]


def get_most_common_floating_point_type(model: torch.nn.Module) -> torch.dtype:
    """Finds the most common floating point data type of the parameters of the model.

    Args:
        model (torch.nn.Module): a model

    Returns:
        torch.dtype: the most common floating point data type of the parameters of the model
    """
    counter = Counter(
        filter(
            lambda dtype: torch.is_floating_point(torch.empty(1, dtype=dtype)),
            (p.dtype for p in model.parameters()),
        )
    )
    if len(counter) == 0:
        return torch.float32
    if len(counter) > 1:
        log.warning(f"The model parameters have more than 1 floating point types: {set(counter.elements())}")
    return counter.most_common(1)[0][0]


def move_tensors_to(
    args: Any,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Any:
    """Assign device and dtype to tensors in a nested structure containing torch.Tensor instances.

    Args:
        args (Any): a nested structure (dict / list / tuple) of torch.Tensor instances.
        device (Optional[torch.device], optional): if provided, moves all tensors to the device. Defaults to None.
        dtype (Optional[torch.dtype], optional): if the dtype is a floating point type, only floating point typed
            tensors in args will be casted to dtype. The behavior is similar when dtype is a signed integer type
            or unsigned integer type. Defaults to None.

    Returns:
        Any: the nested structure of tensors with possibly modified device and dtype.
    """
    if isinstance(args, dict):
        return {key: move_tensors_to(value, device, dtype) for key, value in args.items()}

    if isinstance(args, tuple):
        return tuple(move_tensors_to(x, device, dtype) for x in args)

    if isinstance(args, list):
        return [move_tensors_to(x, device, dtype) for x in args]

    if isinstance(args, torch.Tensor) and dtype is not None:
        is_args_dtype_integral = not args.dtype.is_floating_point and not args.dtype.is_complex
        is_dtype_integral = not dtype.is_floating_point and not dtype.is_complex
        if (is_dtype_integral and is_args_dtype_integral and (args.dtype.is_signed == dtype.is_signed)) or (
            args.dtype.is_floating_point and dtype.is_floating_point
        ):
            args = args.to(dtype)

    if isinstance(args, torch.Tensor) and device is not None:
        args = args.to(device)

    return args


def compare_nested_outputs(
    x: Any,
    y: Any,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
) -> bool:
    """Checks if two nested structure of values share the same nested structure, and if so,
        checks if their value pairs are all close.

    Args:
        x (Any): a nested structure (dict / list / tuple) of values
            (one of Number, torch.Tensor or np.ndarray instances).
        y (Any): another nested structure of values.
        rtol (Optional[float], optional): See the `rtol` parameter in
            https://numpy.org/doc/stable/reference/generated/numpy.allclose.html. Defaults to RTOL_FP16 if the number
            of bits of both x and y are less then 32, RTOL_FP32 otherwise.
        atol (Optional[float], optional): See the `atol` parameter in
            https://numpy.org/doc/stable/reference/generated/numpy.allclose.html. Defaults to ATOL_FP16 if the number
            of bits of both x and y are less then 32, ATOL_FP32 otherwise.
        equal_nan (bool, optional): Whether to compare NaN's as equal. If True, NaN's in a will be considered equal to
            NaN's in b in the output array.. Defaults to False.

    Returns:
        bool: True if x and y shares the same nested structure and their tensors are all close, False otherwise.
    """

    def _as_key_value(x: Union[tuple, list, dict, OrderedDict]) -> Iterable[tuple[Any, Any]]:
        return enumerate(x) if isinstance(x, (tuple, list)) else x.items()

    def _as_path(key: Any) -> str:
        if isinstance(key, str):
            return f'["{key}"]'
        return f"[{key}]"

    def _compare_key_value(l: tuple, r: tuple, path: str) -> bool:
        return (l[0] == r[0]) and _compare(l[1], r[1], path=path + _as_path(l[0]))

    def _compare(lhs: Any, rhs: Any, path: str = "") -> bool:
        if not (lhs.__class__ is rhs.__class__ or isinstance(lhs, rhs.__class__)):
            log.warning(f"Output{path} have different types: {lhs.__class__.__name__}  != {rhs.__class__.__name__}")
            return False

        if isinstance(lhs, (tuple, list, dict, OrderedDict)):
            if len(lhs) != len(rhs):
                log.warning(f"Output{path} have different length: {len(lhs)}  != {len(rhs)}")
                return False
            return all(
                map(
                    lambda l, r: _compare_key_value(l, r, path),
                    _as_key_value(lhs),
                    _as_key_value(rhs),
                )
            )

        if not isinstance(lhs, (torch.Tensor, np.ndarray, Number)):
            log.warning(
                "Expected nested list/tuple/dict/OrderedDict of torch.Tensor/np.ndarray/Number objects, "
                f"but found unsupported type {type(lhs)} while parsing nested structure."
            )
            return False

        if isinstance(lhs, (torch.Tensor, np.ndarray)) and lhs.shape != rhs.shape:
            log.warning(f"Output{path} have different shapes: {typestr(lhs)} != {typestr(rhs)}")
            return False

        are_allclose = allclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=equal_nan)
        if not are_allclose:
            lhs, rhs = map(convert_to_fp_ndarray, (lhs, rhs))
            diff = lhs - rhs
            squared_diff = diff**2
            mse = squared_diff.mean().item()
            max_diff = np.abs(diff).max().item()
            log.warning(
                f"Output{path} of shape {typestr(lhs)} have different values: MSE={mse:3e}, MaxDiff={max_diff:3e}"
            )
        return are_allclose

    return _compare(x, y)


def allclose(
    a: Union[Number, np.ndarray, torch.Tensor],
    b: Union[Number, np.ndarray, torch.Tensor],
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
) -> bool:
    """checks if values are all close.

    Args:
        a (Union[Number, np.ndarray, torch.Tensor]): a value (one of Number, torch.Tensor or np.ndarray instances)
        b (Union[Number, np.ndarray, torch.Tensor]): another value
        rtol (Optional[float], optional): See the `rtol` parameter in
            https://numpy.org/doc/stable/reference/generated/numpy.allclose.html. Defaults to RTOL_FP16 if the number
            of bits of both x and y are less then 32, RTOL_FP32 otherwise.
        atol (Optional[float], optional): See the `atol` parameter in
            https://numpy.org/doc/stable/reference/generated/numpy.allclose.html. Defaults to ATOL_FP16 if the number
            of bits of both x and y are less then 32, ATOL_FP32 otherwise.
        equal_nan (bool, optional): Whether to compare NaN's as equal. If True, NaN's in a will be considered equal to
            NaN's in b in the output array.. Defaults to False.

    Returns:
        bool: True if the values in a and b are all close, False otherwise.
    """
    a, b = map(convert_to_fp_ndarray, (a, b))

    def num_bits(dtype: np.dtype) -> int:
        return dtype.itemsize * 8

    dtype = min(a.dtype, b.dtype, key=num_bits)
    if rtol is None:
        rtol = RTOL_FP32 if dtype == np.dtype("float32") else RTOL_FP16
    if atol is None:
        atol = ATOL_FP32 if dtype == np.dtype("float32") else ATOL_FP16

    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def is_floating_point(dtype: Optional[Union[np.dtype, "TensorProto.DataType"]]) -> bool:
    """Checks if the dtype is a floating point type

    Args:
        dtype (Optional[Union[np.dtype, TensorProto.DataType]]): a dtype

    Returns:
        bool: True if the dtype is a floating point type, False otherwise.
    """
    dtype = get_numpy_type(dtype)
    if dtype is None:
        return False
    return np.issubdtype(dtype, np.floating)


def convert_to_fp_ndarray(x: Union[Number, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Converts a value to floating point numpy array

    Args:
        x (Union[Number, np.ndarray, torch.Tensor]): a value (one of Number, torch.Tensor or np.ndarray instance)

    Returns:
        np.ndarray: the value converted into numpy array
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().resolve_neg().resolve_conj().numpy()
    elif isinstance(x, Number):
        x = np.array(x)

    if not is_floating_point(x.dtype):
        x = x.astype(np.float32)

    return x


def is_onnx_proto_data_external(onnx_proto: ModelProto) -> bool:
    """Checks if given onnx proto does not contain parameters

    Args:
        onnx_proto (ModelProto): onnx proto to check

    Returns:
        bool: True if data are stored in external file False elsewise
    """
    external_count = [i.data_location for i in onnx_proto.graph.initializer].count(TensorProto.EXTERNAL)
    return 2 * external_count >= len(onnx_proto.graph.initializer)


def map_signature(func: Callable, *args: Any, **kwargs: Any) -> list[tuple[str, Any]]:
    """Maps the parameter names of a function to the corresponding values passed in args and kwargs.

    This function returns a list of tuples, where each tuple contains a parameter name and its corresponding value.
    If a parameter name exists in the kwargs dictionary, its value is taken from there. Otherwise, the values are taken
    in order from the args tuple. If there are no values left in args or kwargs, the default value of the parameter
    (if it exists) is used.

    Args:
        func (Callable): Function to inspect.
        args (Any): Positional arguments.
        kwargs (Any): Keyword arguments.

    Returns:
        list[tuple[str, Any]]: List of tuples mapping parameter names to their values.

    Note:
        This function assumes that `args` and `kwargs` match the exact function signature,
        in order and length. If they don't, the result may not be as expected or exceptions might occur.
    """
    sig = inspect.signature(func)
    params = sig.parameters

    mapped = []

    args_iter = iter(args)
    for name, param in params.items():
        if name in kwargs:
            mapped.append((name, kwargs[name]))
        elif args:
            mapped.append((name, next(args_iter, param.default)))
        else:
            mapped.append((name, param.default))

    return mapped


def extract_tensor_shape(
    value_info_or_tensor_type_proto: Union[ValueInfoProto, TypeProto.Tensor]
) -> Optional[list[Union[str, int]]]:
    """Extracts tensor shape information.

    Args:
        value_info_or_tensor_type_proto (Union[ValueInfoProto, TypeProto.Tensor]): protobuf to extract shape from.

    Returns:
        Optional[list[Union[str, int]]]: Extracted shape information if exists None otherwise.
    """
    assert isinstance(value_info_or_tensor_type_proto, (ValueInfoProto, TypeProto.Tensor))

    if isinstance(value_info_or_tensor_type_proto, ValueInfoProto):
        value_info_or_tensor_type_proto = value_info_or_tensor_type_proto.type.tensor_type

    if not value_info_or_tensor_type_proto.shape.dim:
        return None

    tensor_type = value_info_or_tensor_type_proto
    return reduce(lambda acc, cur: acc + [cur.dim_param or cur.dim_value], tensor_type.shape.dim, [])


def extract_input_signature_from_onnx_proto(onnx_proto_or_path: Union[ModelProto, str]) -> ONNXInputSignature:
    """Extracts input signature from onnx proto.

    Args:
        onnx_proto (Union[ModelProto, str]): onnx model or path of an onnx model.

    Returns:
        list[tuple[str, list[Union[str, int]]]]: list of tuples of input tensor name and shape.
    """
    onnx_proto = None
    if isinstance(onnx_proto_or_path, str):
        with open(onnx_proto_or_path, "rb") as f:
            onnx_proto = onnx.load(f, load_external_data=False)
    else:
        onnx_proto = onnx_proto_or_path

    shape = extract_tensor_shape
    return reduce(lambda acc, cur: acc + [(cur.name, shape(cur.type.tensor_type))], onnx_proto.graph.input, [])

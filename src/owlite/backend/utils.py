# ruff: noqa: E741
import json
from collections import Counter, OrderedDict
from collections.abc import Iterable
from numbers import Number
from typing import Any

import numpy as np
import onnx
import torch
from onnx import NodeProto as ONNXNode
from onnx.helper import tensor_dtype_to_np_dtype, tensor_dtype_to_string
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node as FXNode
from torch.fx.node import Target as FXTarget

from ..core.logger import log

try:
    import onnx_graphsurgeon as gs

    AnyNode = FXNode | gs.Node | ONNXNode
except ImportError:
    gs = None
    AnyNode = FXNode | ONNXNode

RTOL_FP16 = np.finfo(np.float16).smallest_normal.item()
RTOL_FP32 = 1.0e-5
ATOL_FP16 = np.finfo(np.float16).eps.item()
ATOL_FP32 = 1.0e-8


# pylint: disable-next=missing-function-docstring
def get_numpy_type(onnx_type: np.dtype | int | None) -> np.dtype | None:
    if onnx_type is None:
        return None

    if isinstance(onnx_type, np.dtype):
        # Already a NumPy type
        return onnx_type

    # For some reason, TENSOR_TYPE_TO_NP_TYPE maps `bfloat16` to `float32`.
    # This obviously breaks things, so we need to treat this as a special case.
    if onnx_type == onnx.TensorProto.BFLOAT16:
        return None
    return tensor_dtype_to_np_dtype(onnx_type)


# pylint: disable-next=missing-function-docstring
def get_onnx_tensor_shape(onnx_tensor: onnx.ValueInfoProto | onnx.TensorProto) -> tuple[int | str, ...] | None:
    if isinstance(onnx_tensor, onnx.TensorProto):
        return tuple(onnx_tensor.dims)
    if not onnx_tensor.type.tensor_type.HasField("shape"):
        return None
    shape = []
    for dim in onnx_tensor.type.tensor_type.shape.dim:
        if dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            shape.append(None)
    return tuple(shape)


# pylint: disable-next=missing-function-docstring
def get_onnx_tensor_dtype(
    onnx_tensor: onnx.ValueInfoProto | onnx.TensorProto,
) -> np.dtype | int:
    if isinstance(onnx_tensor, onnx.TensorProto):
        onnx_type = onnx_tensor.data_type
    else:
        onnx_type = onnx_tensor.type.tensor_type.elem_type

    dtype = get_numpy_type(onnx_type)
    if dtype is not None:
        return dtype

    log.warning(
        f"Could not convert: {tensor_dtype_to_string(onnx_type)} to a corresponding NumPy type. "
        f"The original ONNX type will be preserved. ",
    )
    return onnx_type


# pylint:disable = invalid-name
def nodestr(node: AnyNode | None, show_activations: bool = False) -> str:  # type: ignore[valid-type]
    """Generate the string representation of a node object.

    Args:
        node (AnyNode | None): a node. Must be an instance of one of the types:
            torch.fx.Node or onnx.NodeProto
        show_activations (bool, optional): Only available if node is onnx.NodeProto instance. If True,
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
    if gs is not None and isinstance(node, gs.Node):
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
    """Generate the string representation of the target of a `torch.fx.Node` object.

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


def typestr(tensor: torch.Tensor | np.ndarray) -> str:
    """Generate the MLIR-like string representation of the type of a torch.Tensor or np.ndarray instance.

    Args:
        tensor (torch.Tensor | np.ndarray): a tensor or ndarray

    Returns:
        str: the string representation of the type of the tensor
    """
    return f'{"x".join(map(str, tensor.shape))} ({str(tensor.dtype).removeprefix("torch.")})'


def camel_to_snake(camel_cased_string: str) -> str:
    """Convert given camelCase string to snake_case string.

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
    """Find the most common device where the parameters of the model reside.

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
    """Find the most common floating point data type of the parameters of the model.

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
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Assign device and dtype to tensors in a nested structure containing torch.Tensor instances.

    Args:
        args (Any): a nested structure (dict / list / tuple) of torch.Tensor instances.
        device (torch.device | None, optional): if provided, moves all tensors to the device. Defaults to None.
        dtype (torch.dtype | None, optional): if the dtype is a floating point type, only floating point typed
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
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
) -> bool:
    """Compare nested structures for approximate equality.

    Checks if two nested structure of values share the same nested structure,
    and if so, checks if their value pairs are all close.

    Args:
        x (Any): a nested structure (dict / list / tuple) of values
            (one of Number, torch.Tensor or np.ndarray instances).
        y (Any): another nested structure of values.
        rtol (float | None, optional): See the `rtol` parameter in
            https://numpy.org/doc/stable/reference/generated/numpy.allclose.html. Defaults to RTOL_FP16 if the number
            of bits of both x and y are less then 32, RTOL_FP32 otherwise.
        atol (float | None, optional): See the `atol` parameter in
            https://numpy.org/doc/stable/reference/generated/numpy.allclose.html. Defaults to ATOL_FP16 if the number
            of bits of both x and y are less then 32, ATOL_FP32 otherwise.
        equal_nan (bool, optional): Whether to compare NaN's as equal. If True, NaN's in a will be considered equal to
            NaN's in b in the output array.. Defaults to False.

    Returns:
        bool: True if x and y shares the same nested structure and their tensors are all close, False otherwise.
    """

    def _as_key_value(x: tuple | list | dict | OrderedDict) -> Iterable[tuple[Any, Any]]:
        return enumerate(x) if isinstance(x, tuple | list) else x.items()

    def _as_path(key: Any) -> str:
        if isinstance(key, str):
            return f'["{key}"]'
        return f"[{key}]"

    def _compare_key_value(l: tuple, r: tuple, path: str) -> bool:
        if l[0] != r[0]:
            log.warning(f"An output name has been changed: {r[0]} -> {l[0]}")
            return False
        return _compare(l[1], r[1], path=path + _as_path(l[0]))

    def _compare(lhs: Any, rhs: Any, path: str = "") -> bool:
        if lhs is None and rhs is None:
            return True

        if not (lhs.__class__ is rhs.__class__ or isinstance(lhs, rhs.__class__)):
            log.warning(f"Output{path} have different types: {lhs.__class__.__name__}  != {rhs.__class__.__name__}")
            return False

        if isinstance(lhs, tuple | list | dict | OrderedDict):
            if len(lhs) != len(rhs):
                log.warning(f"The length of output{path} has been changed: {len(rhs)} -> {len(lhs)}")
                return False
            return all(_compare_key_value(l, r, path) for l, r in zip(_as_key_value(lhs), _as_key_value(rhs)))

        if not isinstance(lhs, torch.Tensor | np.ndarray | Number):
            log.warning(
                "Expected nested list/tuple/dict/OrderedDict of torch.Tensor/np.ndarray/Number/None objects, "
                f"but found unsupported type {type(lhs)} while parsing nested structure."
            )
            return False

        if isinstance(lhs, torch.Tensor | np.ndarray) and lhs.shape != rhs.shape:
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
    a: Number | np.ndarray | torch.Tensor,
    b: Number | np.ndarray | torch.Tensor,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
) -> bool:
    """Check if values are all close.

    Args:
        a (Number | np.ndarray | torch.Tensor): a value (one of Number, torch.Tensor or np.ndarray instances)
        b (Number | np.ndarray | torch.Tensor): another value
        rtol (float | None, optional): See the `rtol` parameter in
            https://numpy.org/doc/stable/reference/generated/numpy.allclose.html. Defaults to RTOL_FP16 if the number
            of bits of both x and y are less then 32, RTOL_FP32 otherwise.
        atol (float | None, optional): See the `atol` parameter in
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


def is_floating_point(dtype: np.dtype | int | None) -> bool:
    """Check if the dtype is a floating point type.

    Args:
        dtype (np.dtype | int | None): a numpy data type or an integer as in [onnx.TensorProto.DataType](https://github.com/dmlc/tensorboard/blob/d36e8e921cdd5306c7e2535adbc0fe45be47ceed/tensorboard/src/onnx.proto#L182-L204)

    Returns:
        bool: True if the dtype is a floating point type, False otherwise.
    """
    dtype = get_numpy_type(dtype)
    if not isinstance(dtype, np.dtype):
        return False
    return np.issubdtype(dtype, np.floating)


def convert_to_fp_ndarray(x: Number | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a value to floating point numpy array.

    Args:
        x (Number | np.ndarray | torch.Tensor): a value (one of Number, torch.Tensor or np.ndarray instance)

    Returns:
        np.ndarray: the value converted into numpy array
    """
    if isinstance(x, torch.Tensor):
        x = x.numpy(force=True)
    elif isinstance(x, Number):
        x = np.array(x)

    if not is_floating_point(x.dtype):  # type: ignore
        x = x.astype(np.float32)  # type: ignore

    return x  # type: ignore

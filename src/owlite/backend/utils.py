# ruff: noqa: E741
import json
from numbers import Number

import numpy as np
import onnx
import torch
from onnx import NodeProto as ONNXNode
from onnx.helper import tensor_dtype_to_np_dtype, tensor_dtype_to_string
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node as FXNode
from torch.fx.node import Target as FXTarget
from torch.utils._pytree import PyTree, tree_flatten

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
def get_numpy_type(
    onnx_tensor_or_data_type: onnx.ValueInfoProto | onnx.TensorProto | np.dtype | int | None,
) -> np.dtype | None:
    if isinstance(onnx_tensor_or_data_type, np.dtype | None):
        return onnx_tensor_or_data_type

    if isinstance(onnx_tensor_or_data_type, onnx.ValueInfoProto):
        data_type = onnx_tensor_or_data_type.type.tensor_type.elem_type
    elif isinstance(onnx_tensor_or_data_type, onnx.TensorProto):
        data_type = onnx_tensor_or_data_type.data_type
    else:
        data_type = onnx_tensor_or_data_type

    try:
        return tensor_dtype_to_np_dtype(data_type)
    except KeyError as e:
        log.warning(
            f"ONNX data type {onnx.helper.tensor_dtype_to_string(data_type)} "
            f"cannot be converted to a numpy data type. ({e})"
        )  # UX
        return None


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


def compare_pytrees(
    x: PyTree,
    y: PyTree,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
    compare_tree_specs: bool = False,
) -> bool:
    """Compare all the values in given python object trees.

    Args:
        x (PyTree): a python object tree that whose leaves are one of Number, torch.Tensor or np.ndarray objects.
        y (PyTree): another such python object tree.
        rtol (float | None, optional): See the `rtol` parameter in
            https://numpy.org/doc/stable/reference/generated/numpy.allclose.html. Defaults to RTOL_FP16 if the number
            of bits of both x and y are less then 32, RTOL_FP32 otherwise.
        atol (float | None, optional): See the `atol` parameter in
            https://numpy.org/doc/stable/reference/generated/numpy.allclose.html. Defaults to ATOL_FP16 if the number
            of bits of both x and y are less then 32, ATOL_FP32 otherwise.
        equal_nan (bool, optional): Whether to compare NaN's as equal. If True, NaN's in a will be considered equal to
            NaN's in b in the output array. Defaults to False.
        compare_tree_specs (bool, optional): Whether to compare the spec of the given object trees. Defaults to False.

    Returns:
        bool: True if all values in x and y are approximately equal, False otherwise.
    """
    flat_x, spec_x = tree_flatten(x)
    flat_y, spec_y = tree_flatten(y)
    have_same_specs = spec_x == spec_y
    if not have_same_specs:
        log.warning(f"Comparing objects with different pytree specs:\n{spec_x}\n{spec_y}")
    are_comparable = have_same_specs if compare_tree_specs else len(flat_x) == len(flat_y)
    return are_comparable and all(
        allclose(item_x, item_y, rtol=rtol, atol=atol, equal_nan=equal_nan) for item_x, item_y in zip(flat_x, flat_y)
    )


def allclose(
    a: Number | np.ndarray | torch.Tensor,
    b: Number | np.ndarray | torch.Tensor,
    *,
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

    if not is_floating_point(x.dtype):
        x = x.astype(np.float32)

    return x

import os
from collections import Counter, OrderedDict
from enum import Enum
from typing import Any, Callable, Optional, Union, cast

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import ModelProto, TensorProto

from ...owlite_core.logger import log
from ..config import ONNX_TRANSFORM_MAXIMUM_ITERATION
from ..utils import get_numpy_type, is_floating_point, nodestr
from .export_with_external_data import export_with_external_data
from .fold_constants import fold_constants  # type: ignore
from .onnx_op import ONNXOp

OnnxTransform = Callable[[gs.Graph], gs.Graph]
ONNX_TRANSFORMS: dict[str, OnnxTransform] = {}
Tensor = Union[gs.Constant, gs.Variable]


def apply_onnx_transforms(onnx_proto: ModelProto, output_path: Optional[str] = None, **kwargs: Any) -> ModelProto:
    """Applies all transformations registered in this file.

    Args:
        onnx_proto (ModelProto): the ONNX model proto to apply transformations.
        output_path (Optional[str], optional): the output path in string. If provided, runs the ModelProto will be
            written with external data after the transformations (required for large models > 2GB). Defaults to None.

    Returns:
        ModelProto: the transformed ONNX model proto.
    """
    graph = gs.import_onnx(onnx_proto)
    for name, transform in ONNX_TRANSFORMS.items():
        log.debug(f"Applying ONNX transform: {name}")
        graph = transform(graph)
    graph.toposort()
    graph = fold_constants(graph)
    graph.cleanup()
    if output_path is None:
        return gs.export_onnx(graph)
    export_with_external_data(graph, output_path, **kwargs)
    return onnx.load(output_path)


def register_onnx_transform(transform: OnnxTransform) -> OnnxTransform:
    """Registers a ONNX transform globally. Note that the registration order matters.

    Use this function as a decorator to register your custom ONNX transform. For example:
    @register_onnx_transform
    def do_something_on_onnx_graph(graph: gs.Graph) -> gs.Graph:
        ...
    """
    name = transform.__name__
    if name in ONNX_TRANSFORMS:
        log.debug_warning(f"Overwriting existing ONNX transform: {name}")
    ONNX_TRANSFORMS[name] = transform
    return transform


@register_onnx_transform
def fold_trilu_constants(graph: gs.Graph) -> gs.Graph:
    """Folds Trilu ops if constant-foldable. Note that this transformation is a workaround for the missing support for
    the Trilu op in onnx-runtime

    Args:
        graph (gs.Graph): a ONNX graph.

    Returns:
        gs.Graph: the ONNX graph with constant-foldable Trilu ops removed.
    """
    for node in graph.nodes:
        if node.op == "Trilu":
            input_node = input_node_of(node, 0)
            if input_node is None or "value" not in input_node.attrs:
                continue
            input_values: np.ndarray = input_node.attrs["value"].values

            k_node = input_node_of(node, 1)
            if k_node is None:
                k_value = 0
            elif "value" in k_node.attrs:
                k_value = k_node.attrs["value"].values.item()
            else:
                continue

            folded_values: np.ndarray = np.tril(input_values, k_value)

            output_tensor: gs.Variable = node.outputs[0]
            output_tensor.inputs.clear()
            output_const = gs.Constant(name=f"{node.name}_const", values=folded_values)

            const_node = gs.Node(
                op="Constant",
                name=f"{node.name}_folded",
                attrs=OrderedDict([("value", output_const)]),
                outputs=[output_tensor],
            )

            graph.nodes.remove(node)
            graph.nodes.append(const_node)

            log.debug(f"Replaced {nodestr(node)} by {nodestr(const_node)}")

    graph.cleanup()
    graph.toposort()
    return graph


@register_onnx_transform
def eliminate_nop_dropouts(graph: gs.Graph) -> gs.Graph:
    """Eliminates all Dropout ops with no effect.

    Args:
        graph (gs.Graph): a ONNX graph.

    Returns:
        gs.Graph: the ONNX graph with meaningless Dropout ops removed.
    """
    for node in graph.nodes:
        remove_if_dropout_op_with_ratio_zero(node, graph)

    graph.cleanup()
    graph.toposort()
    return graph


@register_onnx_transform
def eliminate_nop_casts(graph: gs.Graph) -> gs.Graph:
    """Eliminates all Cast ops with no effect.

    Args:
        graph (gs.Graph): a ONNX graph.

    Returns:
        gs.Graph: the ONNX graph with meaningless Cast ops removed.
    """
    for node in graph.nodes:
        remove_if_cast_op_with_no_effect(node, graph)

    graph.cleanup()
    graph.toposort()
    return graph


@register_onnx_transform
def synchronize_floating_point_types(graph: gs.Graph) -> gs.Graph:
    """Synchronizes all floating points types used in the graph as the most common one.

    Args:
        graph (gs.Graph): a ONNX graph.

    Returns:
        gs.Graph: the ONNX graph with only one floating point type.
    """
    floating_point_dtypes: list[np.dtype] = [*filter(is_floating_point, (t.dtype for t in graph.tensors().values()))]
    counter = Counter(floating_point_dtypes)
    log.debug(f"Counts of floating point types: {counter}")

    if len(counter) == 0:
        log.debug("No tensor with floating point type found in the graph")
        return graph

    class FloatingPointSyncType(Enum):
        """How to synchronize the floating point types within a model"""

        FP32 = 0
        FP16 = 1
        MOST_COMMON = 2

    sync_type_name = os.getenv("OWLITE_FLOATING_POINT_SYNC_TYPE", "MOST_COMMON")
    try:
        sync_type = FloatingPointSyncType[sync_type_name]
    except KeyError as e:
        log.error(
            f"Invalid value {sync_type_name} given to the environment variable 'OWLITE_FLOATING_POINT_SYNC_TYPE'. "
            f"It must be one of {','.join(t.name for t in FloatingPointSyncType)}"
        )
        raise e

    dtype: np.dtype
    match sync_type:
        case FloatingPointSyncType.FP32:
            dtype = np.dtype("float32")
        case FloatingPointSyncType.FP16:
            dtype = np.dtype("float16")
        case FloatingPointSyncType.MOST_COMMON:
            dtype = counter.most_common(1)[0][0]
            log.debug(f"Most common floating point type: {dtype}")

    if len(counter) > 1:
        log.warning(
            "More than one floating point types are used in the graph ("
            f'{", ".join(f"{value} tensors of type {key.name}" for key, value in OrderedDict(counter).items())}). '
            f"Will use {dtype.name} as 'OWLITE_FLOATING_POINT_SYNC_TYPE' was set to {sync_type.name}"
        )

    for node in graph.nodes:
        cast_input_fp_tensors_of(node, dtype)
        cast_output_fp_tensors_of(node, dtype)

    return eliminate_nop_casts(graph)


@register_onnx_transform
def fold_nodes_after_conv(graph: gs.Graph) -> gs.Graph:
    """Fold foldable element-wise operations after convolution into convolution's weight and bias.

    Args:
        graph (gs.Graph): a ONNX graph.

    Returns:
        gs.Graph: the transformed ONNX graph
    """

    # pylint: disable=too-many-statements

    def _get_constant_or_none(tensor: Tensor) -> Optional[gs.Constant]:
        if isinstance(tensor, gs.Constant):
            return tensor

        if len(tensor.inputs) != 1:
            return None

        if tensor.inputs[0].op == "Constant" and isinstance(tensor.inputs[0].attrs.get("value"), gs.Constant):
            return tensor.inputs[0].attrs.get("value")

        if tensor.inputs[0].op == "Identity":
            return _get_constant_or_none(tensor.i())

        return None

    def _get_constant_conv_params(
        conv_node: gs.Node,
    ) -> Optional[tuple[gs.Constant, Optional[gs.Constant], Optional[gs.Constant]]]:
        if conv_node.op != "Conv":
            raise ValueError(f"Expected a `Conv` operation but received `{conv_node.op}`")

        # weight is required field for conv
        conv_weight = conv_node.inputs[1]

        # bias is optional input for conv
        conv_bias = conv_node.inputs[2] if len(conv_node.inputs) == 3 else None

        # we do not consider zero point for now
        quantizer_step_size = None

        if isinstance(conv_weight, gs.Variable) and get_defining_op_type(conv_weight) == "DequantizeLinear":
            dequantize_node = conv_weight.inputs[0]

            if get_defining_op_type(dequantize_node.inputs[0]) != "QuantizeLinear":
                # parent node of DequantizeLinear is not QuantizeLinear
                return None

            quantizer_step_size = dequantize_node.inputs[1]
            if len(quantizer_step_size.outputs) != 2:
                # quantizer step size used somewhere else than current quantizers,
                # note that QuantizeLinear and DequantizeLinear from same quantizer shares the same tensor
                return None

            quantize_node = dequantize_node.inputs[0].inputs[0]
            if quantize_node.inputs[1] is not quantizer_step_size:
                # QuantizeLinear does not share the same tensor as step size with DequantizeLinear
                return None

            quantizer_step_size = _get_constant_or_none(quantizer_step_size)
            if quantizer_step_size is None and quantize_node.i(1).op == "Abs":
                # Abs operation is inserted for CLQ
                quantizer_step_size = _get_constant_or_none(quantize_node.i(1).inputs[0])

            if quantizer_step_size is None:
                # quantizer step size is variable
                return None

            conv_weight = quantize_node.inputs[0]

        conv_weight = _get_constant_or_none(conv_weight)
        if conv_weight is None:
            return None

        if conv_bias is not None:
            conv_bias = _get_constant_or_none(conv_bias)
            if conv_bias is None:
                return None

        assert (
            isinstance(conv_weight, gs.Constant)
            and isinstance(conv_bias, (gs.Constant, type(None)))
            and isinstance(quantizer_step_size, (gs.Constant, type(None)))
        )

        return conv_weight, conv_bias, quantizer_step_size

    def _is_narrow_range_quantization(conv_weight: gs.Constant, step_size: gs.Constant) -> bool:
        assert conv_weight.values.ndim == 4
        assert step_size.values.ndim in (0, 1)

        step_size_value = np.atleast_1d(step_size.values)[:, np.newaxis, np.newaxis, np.newaxis]

        return np.greater(np.round(np.min(conv_weight.values / step_size_value)), -128)

    def _get_constant_param_to_fold(node_to_fold: gs.Node) -> Optional[gs.Constant]:
        parameter_to_fold = [_get_constant_or_none(tensor) for tensor in node_to_fold.inputs]
        parameter_to_fold = [tensor for tensor in parameter_to_fold if tensor is not None]

        if len(parameter_to_fold) == 1:
            return parameter_to_fold[0]

        return None

    # pylint: disable-next=too-many-return-statements
    def _is_foldable(conv_output_tensor: gs.Variable) -> bool:
        # convolution output is dependant to other node than conv or convolution output is used more than once
        if len(conv_output_tensor.inputs) != 1 or len(conv_output_tensor.outputs) != 1:
            return False

        conv_node: gs.Node = conv_output_tensor.inputs[0]
        node_to_fold: gs.Node = conv_output_tensor.outputs[0]

        # only the element-wise operations are foldable, and we cannot fold Div(param, Conv(x, w, b))
        if node_to_fold.op not in ("Add", "Sub", "Mul", "Div") or (
            node_to_fold.op == "Div" and node_to_fold.inputs[1] is conv_output_tensor
        ):
            return False

        # all involved tensors should be constant
        parameter_to_fold = _get_constant_param_to_fold(node_to_fold)
        conv_node_params = _get_constant_conv_params(conv_node)
        if parameter_to_fold is None or conv_node_params is None:
            return False

        conv_weight, conv_bias, quantizer_step_size = conv_node_params

        # disclaimer: we now only consider 2d convolution, with this removed, conditions below should be revisited
        if conv_weight.values.ndim != 4 or (conv_bias is not None and conv_bias.values.ndim != 1):
            return False

        # only 0,1 or 4-dimensional parameters are foldable for 2d convolution
        if parameter_to_fold.values.ndim not in (0, 1, 4):
            return False

        # cannot broadcast parameter to convolution output channel dimension
        if parameter_to_fold.values.ndim != 0 and parameter_to_fold.values.size != conv_weight.values.shape[0]:
            return False

        # cannot broadcast parameter to convolution output channel dimension
        if parameter_to_fold.values.ndim == 4 and parameter_to_fold.values.shape[1] != conv_weight.values.shape[0]:
            return False

        if quantizer_step_size is not None:
            # cannot fold negative values into quantized weight when -128 bin is not empty
            if (
                (node_to_fold.op in ("Mul", "Div") and np.min(parameter_to_fold.values) < 0)
                or (node_to_fold.op == "Sub" and node_to_fold.inputs[1] is conv_output_tensor)
                and not _is_narrow_range_quantization(conv_weight, quantizer_step_size)
            ):
                return False

            # cannot broadcast parameter to fold to per-tensor quantization step size
            if parameter_to_fold.values.size != 1 and quantizer_step_size.values.size == 1:
                return False

        return True

    def _fold(conv_node: gs.Node, node_to_fold: gs.Node) -> None:
        def _fold_mul_or_div(conv_node: gs.Node, mul_or_div_node: gs.Node) -> None:
            assert conv_node.op == "Conv" and mul_or_div_node.op in ("Mul", "Div")

            conv_params = _get_constant_conv_params(conv_node)
            assert conv_params is not None
            conv_weight, conv_bias, quantizer_step_size = conv_params

            param_to_fold = _get_constant_param_to_fold(mul_or_div_node)
            assert param_to_fold is not None
            value_to_fold: np.ndarray = (
                param_to_fold.values if mul_or_div_node.op == "Mul" else (param_to_fold.values**-1)
            )

            value_to_fold = np.broadcast_to(value_to_fold.squeeze(), conv_weight.values.shape[0])

            if conv_bias is not None:
                conv_bias.values *= value_to_fold

            if quantizer_step_size is not None:
                if quantizer_step_size.values.size == 1:
                    assert np.all(value_to_fold == value_to_fold[0])
                    quantizer_step_size.values *= np.abs(value_to_fold)[0]

                else:
                    quantizer_step_size.values *= np.abs(value_to_fold)

            target_shape = (
                value_to_fold.shape[0],
                1,
                1,
                1,
            )
            conv_weight.values *= value_to_fold.reshape(target_shape)

            conv_node.outputs = mul_or_div_node.outputs
            mul_or_div_node.outputs.clear()

        def _fold_add_or_sub(conv_node: gs.Node, add_or_sub_node: gs.Node) -> None:
            assert conv_node.op == "Conv" and add_or_sub_node.op in ("Add", "Sub")

            conv_params = _get_constant_conv_params(conv_node)
            assert conv_params is not None
            conv_weight, conv_bias, _ = conv_params

            param_to_fold = _get_constant_param_to_fold(add_or_sub_node)
            assert param_to_fold is not None
            value_to_fold: np.ndarray = param_to_fold.values

            value_to_fold = np.broadcast_to(value_to_fold.squeeze(), conv_weight.values.shape[0])

            if add_or_sub_node.op == "Sub":
                if add_or_sub_node.inputs[1] is conv_node.outputs[0]:  # Sub(param, Conv(x, w, b))
                    conv_weight.values = -conv_weight.values
                    if conv_bias is not None:
                        conv_bias.values = -conv_bias.values

                else:  # Sub(Conv(x, w, b), param)
                    value_to_fold = -value_to_fold

            if conv_bias is not None:
                conv_bias.values += value_to_fold

            else:
                new_bias = gs.Constant(param_to_fold.name, value_to_fold, param_to_fold.data_location)
                conv_node.inputs.append(new_bias)

            conv_node.outputs = add_or_sub_node.outputs
            add_or_sub_node.outputs.clear()

        if node_to_fold.op in ("Mul", "Div"):
            _fold_mul_or_div(conv_node, node_to_fold)

        elif node_to_fold.op in ("Add", "Sub"):
            _fold_add_or_sub(conv_node, node_to_fold)

        else:  # for future extensibility, we might be able to fold more operations
            raise NotImplementedError()

    for node in graph.nodes:
        if node.op == "Conv":
            i = 0
            while i < ONNX_TRANSFORM_MAXIMUM_ITERATION:
                if _is_foldable(node.outputs[0]):
                    log.debug(f"Folding {nodestr(node.outputs[0].outputs[0])} into {nodestr(node)}")
                    _fold(node, node.outputs[0].outputs[0])
                    i += 1
                    continue

                break

    return graph


@register_onnx_transform
def eliminate_nop_reformatting_sequences(graph: gs.Graph) -> gs.Graph:
    """Eliminate meaningless reformatting sequences
    (e.g. Flatten->Reshape with identical input and output shapes)

    Args:
        graph (gs.Graph): a ONNX graph.

    Returns:
        gs.Graph: the transformed ONNX graph
    """
    reformatting_ops = ("Reshape", "Flatten", "Squeeze", "Unsqueeze")
    for node in graph.nodes:
        if node.op not in reformatting_ops:
            continue
        reformatting_sequence = [node]
        log.debug(f"Starting with {nodestr(node)}")
        while True:
            child = reformatting_sequence[-1]
            if not ((parent := get_defining_node(child.inputs[0])) is not None and parent.op in reformatting_ops):
                log.debug(f"Break: parent {nodestr(parent)} is not a reformatting node")
                break

            reformatting_sequence.append(parent)
            log.debug(f"Appended {nodestr(parent)}")
            if parent.inputs[0].shape == reformatting_sequence[0].outputs[0].shape:
                log.debug("Early exit: parent input shape matched")
                break
        if not (
            len(reformatting_sequence) > 1
            and (
                (the_output := reformatting_sequence[0].outputs[0]).shape
                == (the_input := reformatting_sequence[-1].inputs[0]).shape
            )
        ):
            log.debug("No foldable reformatting sequence found")
            continue
        log.debug(f"Found foldable reformatting sequence: {[*map(nodestr, reformatting_sequence)]}")
        replace_all_uses(the_output, the_input)
    return graph


# pylint: disable=missing-function-docstring
def remove_if_dropout_op_with_ratio_zero(node: gs.Node, graph: gs.Graph) -> None:
    if node.op != "Dropout":
        return

    if len(node.inputs) == 1:
        return

    if isinstance(node.inputs[1], gs.Variable):
        ratio_input_node = input_node_of(node, 1, 0)

        if ratio_input_node is None or "value" not in ratio_input_node.attrs:
            return

        if ratio_input_node.attrs["value"].values.item() != 0.0:
            return

    else:
        if node.inputs[1].values.item() != 0.0:
            return

    remove_if_has_unique_non_optional_input_and_unique_used_output(node, graph)


def remove_if_cast_op_with_no_effect(node: gs.Node, graph: gs.Graph) -> None:
    if node.op != "Cast":
        return

    if "to" not in node.attrs:
        log.debug_warning(f'Missing required attribute "to" in {nodestr(node)}')
        return

    if len(node.inputs) != 1:
        log.debug_warning(f"{nodestr(node)} must have 1 input but {len(node.inputs)} found: {node.inputs}")
        return

    if len(node.outputs) != 1:
        log.debug_warning(f"{nodestr(node)} must have 1 output but {len(node.outputs)} found: {node.outputs}")
        return

    the_input = node.inputs[0]
    the_output = node.outputs[0]
    data_type = cast(TensorProto.DataType, node.attrs["to"])
    dtype = get_numpy_type(data_type)
    if not isinstance(dtype, np.dtype):
        log.debug_warning(
            f'Failed to convert attribute "to": {TensorProto.DataType.Name(data_type)}'
            " of {nodestr(node)} into numpy type"
        )
        return

    if the_input.dtype in (dtype, the_output.dtype):
        remove_if_has_unique_non_optional_input_and_unique_used_output(node, graph)


def cast_input_fp_tensors_of(node: gs.Node, dtype: np.dtype) -> None:
    onnx_op = ONNXOp(node.op)
    if not onnx_op.is_valid:
        log.debug_warning(f"Unsupported ONNXOp: {node.op}")
        return

    for idx, tensor in enumerate(node.inputs):
        if dtype not in onnx_op.schema.i(idx).allowed_types:
            continue

        if not tensor.inputs and is_floating_point(tensor.dtype):
            set_dtype(tensor, dtype)
            continue

        for input_node in tensor.inputs:
            if input_node.op in ("Constant", "ConstantOfShape") and "value" in input_node.attrs:
                constant: gs.Constant = input_node.attrs["value"]
                if not is_floating_point(constant.dtype) or constant.dtype == dtype:
                    continue
                set_dtype(constant, dtype)


def cast_output_fp_tensors_of(node: gs.Node, dtype: np.dtype) -> None:
    onnx_op = ONNXOp(node.op)
    if not onnx_op.is_valid:
        log.debug_warning(f"Unsupported ONNXOp: {node.op}")
        return

    for idx, tensor in enumerate(node.outputs):
        if dtype not in onnx_op.schema.o(idx).allowed_types:
            continue

        if not is_floating_point(tensor.dtype) or tensor.dtype == dtype:
            continue

        set_dtype(tensor, dtype)


# pylint: disable=protected-access
def set_dtype(tensor: gs.Tensor, dtype: np.dtype) -> None:
    original_repr = f"{tensor}"
    if isinstance(tensor, gs.Variable):
        tensor.dtype = dtype
    elif isinstance(tensor, gs.Constant):
        tensor._values = tensor.values.astype(dtype)
    else:
        log.debug_warning(f"{tensor} is neither a constant nor a variable")
        return
    log.debug(f"{original_repr} -> {tensor}")


def remove_if_has_unique_non_optional_input_and_unique_used_output(node: gs.Node, graph: gs.Graph) -> None:
    used_outputs = [*filter(lambda t: len(t.outputs) > 0 or t in graph.outputs, node.outputs)]
    if len(used_outputs) != 1:
        log.debug_warning(f"Not removing {nodestr(node)} as it doesn't have unique used outputs: {used_outputs}")
        return

    onnx_op = ONNXOp(node.op)
    if not onnx_op.is_valid:
        log.debug_warning(f"Unsupported ONNXOp: {node.op}")
        return

    non_optional_inputs = [node.inputs[idx] for idx in range(len(node.inputs)) if not onnx_op.schema.i(idx).is_optional]

    if len(non_optional_inputs) != 1:
        log.debug_warning(
            f"Not removing {nodestr(node)} as it doesn't have unique non-optional inputs: {non_optional_inputs}"
        )
        return

    the_input = non_optional_inputs[0]
    the_output = used_outputs[0]

    if the_output in graph.outputs:
        graph.outputs = [the_input if t is the_output else t for t in graph.outputs]
        log.debug(f"Replaced the graph output {the_output} by {the_input}")
        the_input.name = the_output.name
    else:
        child_nodes = [*the_output.outputs]
        log.debug(f"Child nodes of {nodestr(node)}: {[*map(nodestr, child_nodes)]}")
        for child_node in child_nodes:
            child_node.inputs = [the_input if t is the_output else t for t in child_node.inputs]
            log.debug(f"Modified child node: {nodestr(child_node, True)}")
            if child_node not in the_input.outputs:
                the_input.outputs.append(child_node)
                log.debug(f"Added {nodestr(child_node)} to the outputs of {the_input}")

    node.outputs.clear()
    log.debug(f"Removed {nodestr(node, True)}")


def input_node_of(node: gs.Node, tensor_idx: int = 0, producer_idx: int = 0) -> Optional[gs.Node]:
    if len(node.inputs) > tensor_idx and len(node.inputs[tensor_idx].inputs) > producer_idx:
        return node.i(tensor_idx, producer_idx)
    return None


def output_node_of(node: gs.Node, tensor_idx: int = 0, consumer_idx: int = 0) -> Optional[gs.Node]:
    if len(node.outputs) > tensor_idx and len(node.outputs[tensor_idx].outputs) > consumer_idx:
        return node.o(consumer_idx, tensor_idx)
    return None


def get_defining_op_type(tensor: Tensor) -> Optional[str]:
    """Get defining node's operation type, return None if tensor has two or more defining ops

    Args:
        tensor (Tensor): tensor to examine

    Returns:
        Optional[str]: ONNX operation type in string or None if cannot specify single unique defining op

    """
    if len(tensor.inputs) != 1:
        return None

    if isinstance(tensor, gs.Constant):
        return None

    return tensor.inputs[0].op


def get_defining_node(tensor: Tensor) -> Optional[gs.Node]:
    if tensor.inputs:
        return tensor.inputs[0]
    return None


def replace_all_uses(existing_tensor: Tensor, new_tensor: Tensor) -> None:
    for node in existing_tensor.outputs:
        i = node.inputs.index(existing_tensor)
        node.inputs[i] = new_tensor

# pylint: disable=C0116, R0914, R0912, R0915, R1702

from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Union

import numpy as np
import onnx_graphsurgeon as gs
from onnx import ModelProto
from onnx.shape_inference import infer_shapes

from ...options import DynamicAxisOptions
from ..signature import Signature
from ..utils import nodestr


def dynamize(onnx_proto: ModelProto, options: DynamicAxisOptions) -> ModelProto:
    """Dynamizes given ONNX proto with given dynamic dimension setting.

    Args:
        onnx_proto (ModelProto): ONNX model proto to dynamize.
        dynamic_dims (DynamicAxisOptions): Dynamic axis setting.

    Raises:
        ValueError: When dynamic ONNX proto is given.
        NotImplementedError: When dynamizing ONNX proto with reshapes with dynamic target shape is attempted.
        RuntimeError: When attempt to dynamize given ONNX proto has failed.

    Returns:
        ModelProto: Dynamized ONNX proto.
    """

    graph = gs.import_onnx(onnx_proto)
    input_signature = Signature.from_onnx(graph)

    if isinstance(input_signature, Signature) and input_signature.is_dynamic:
        raise ValueError("Dynamic ONNX proto given")

    remove_neg_ones_from_reshape_ops(graph)

    for input_tensor in graph.inputs:
        if not isinstance(input_tensor, gs.Variable) or input_tensor.shape is None:
            continue
        axis_options = options.get(input_tensor.name)
        if axis_options is None:
            continue

        dynamic_axis = axis_options.axis
        shape = tuple(-1 if isinstance(s, str) else s for s in input_tensor.shape)
        input_tensor.shape = tuple("N" if i == dynamic_axis else s for i, s in enumerate(input_tensor.shape))
        propagate_dynamic_shape(input_tensor, shape, dynamic_axis)

    dynamized_proto = gs.export_onnx(graph)
    dynamized_proto.graph.ClearField("value_info")
    for output in dynamized_proto.graph.output:
        output.type.tensor_type.ClearField("shape")

    dynamized_proto = infer_shapes(dynamized_proto, check_type=True, strict_mode=True, data_prop=True)

    if not all(
        len([dim.dim_param for dim in output.type.tensor_type.shape.dim if not dim.dim_value]) == 1
        for output in dynamized_proto.graph.output
    ):
        raise RuntimeError("Failed to dynamize given ONNX proto")

    return dynamized_proto


def remove_neg_ones_from_reshape_ops(graph: gs.Graph) -> None:
    for node in graph.nodes:
        if node.op == "Reshape":
            if isinstance(node.inputs[1], gs.Variable):
                raise NotImplementedError("Dynamizing reshapes with dynamic target shape is not supported yet")

            if -1 in node.inputs[1].values:
                node.inputs[1] = gs.Constant(
                    f"{node.name}_target_shape", np.array(node.outputs[0].shape), node.inputs[1].data_location
                )


@dataclass
class DynamicAxisPropagator:
    """The object holding information for propagating dynamic axis"""

    dynamic_axis: int
    original_size: int
    shape: tuple[Union[int, str], ...]
    node: gs.Node


def propagate_dynamic_shape(input_tensor: gs.Variable, shape: tuple[int, ...], dynamic_axis: int) -> None:
    visited: set[str] = set()
    handled_tensors: set[str] = set()

    # pylint: disable-next=too-many-branches
    def propagate(propagators: list[DynamicAxisPropagator]) -> None:
        while len(propagators) > 0:
            propagator = propagators.pop()

            if any(isinstance(s, str) for s in propagator.shape):
                raise ValueError(
                    "Handling `owlite.onnx.dynamize` cannot handle tensors with shape variable. "
                    f"The problematic shape was {propagator.shape}, "
                    f"which is the shape of one of input tensors of the node {nodestr(propagator.node, True)}."
                )

            input_dynamic_dim = propagator.dynamic_axis
            original_size = propagator.original_size
            input_shape = tuple(s for s in propagator.shape if isinstance(s, int))
            node = propagator.node

            if node.name in visited:
                continue

            output_dynamic_dim = input_dynamic_dim

            if node.op == "Reshape":
                if node.inputs[1].name in handled_tensors:
                    index = np.where(node.inputs[1].values == -1)[0].item()
                    output_dynamic_dim = index

                else:
                    target_shape = node.inputs[1].values
                    elements_til_dynamic_dimension = reduce(lambda x, y: x * y, input_shape[: input_dynamic_dim + 1])

                    acc = 1
                    for i, dim in enumerate(target_shape):
                        if dim == original_size:
                            output_dynamic_dim = i
                            break

                        acc = acc * dim
                        if acc >= elements_til_dynamic_dimension:
                            output_dynamic_dim = i
                            break

                    target_shape[output_dynamic_dim] = -1
                    handled_tensors.add(node.inputs[1].name)

            elif node.op == "Resize":
                axes: Sequence[int] = node.attrs.get("axes")
                if axes is None or len(axes) == len(node.outputs[0].shape):
                    scales = node.inputs[2]

                    if scales.name:
                        assert len(node.inputs) < 4 or not node.inputs[3].name

                    else:
                        sizes = node.inputs[3]
                        if isinstance(sizes, gs.Constant) and sizes.name:
                            target_shape = sizes.values

                            if target_shape[input_dynamic_dim] != node.outputs[0].shape[input_dynamic_dim]:
                                raise ValueError("Dynamic ONNX proto given")

                            node.inputs[2] = gs.Constant(
                                f"{node.name}_scale",
                                np.array(target_shape / node.inputs[0].shape, dtype=np.float32),
                                node.inputs[3].data_location,
                            )
                            node.inputs[3] = gs.Variable.empty()

                elif input_dynamic_dim in axes:
                    raise ValueError("Dynamic ONNX proto given")

            elif node.op == "Transpose":
                target_permutation: Sequence[int] = node.attrs.get("perm")
                output_dynamic_dim = target_permutation.index(input_dynamic_dim)

            elif node.op == "Gather":
                axis = node.attrs.get("axis")
                if axis is not None:
                    if axis == input_dynamic_dim:
                        raise ValueError("Dynamic ONNX proto given")

                    should_shift = 1 if axis < input_dynamic_dim else 0
                    output_dynamic_dim = input_dynamic_dim - should_shift

            for output in node.outputs:
                propagators.extend(
                    DynamicAxisPropagator(output_dynamic_dim, original_size, output.shape[:], child)
                    for child in output.outputs
                )

            visited.add(node.name)

    propagate(
        [
            DynamicAxisPropagator(
                dynamic_axis=dynamic_axis,
                original_size=shape[dynamic_axis],
                shape=tuple(shape),
                node=node,
            )
            for node in input_tensor.outputs
        ]
    )

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import onnx_graphsurgeon as gs
from onnx import ModelProto
from onnx.shape_inference import infer_shapes

from ...logger import log
from ..utils import extract_tensor_shape


@dataclass
class DynamicSetting:
    """Dynamic dimension setting for a model input tensor."""

    shape: list[int]
    dim: int
    min: int
    max: int
    opt: int

    @property
    def trt_shapes(self) -> dict[str, list[int]]:  # pylint: disable=missing-function-docstring
        keys = ["min", "max", "opt"]

        return dict(
            zip(
                keys,
                ([size if d != self.dim else getattr(self, key) for d, size in enumerate(self.shape)] for key in keys),
            )
        )


@dataclass
class DynamicDimensions:
    """Dynamic dimension settings."""

    dim_size: int
    settings: dict[str, DynamicSetting]

    @property
    def dynamic_input_names(self) -> list[str]:  # pylint: disable=missing-function-docstring
        return list(self.settings.keys())

    def get(self, name: str) -> Optional[DynamicSetting]:
        """Gets dynamic setting to be applied to tensor with given name.

        Args:
            name (str): Name of a model input tensor.

        Returns:
            Optional[DynamicSetting]: Setting to be applied if setting for given name exists.
        """
        return self.settings.get(name)


# pylint: disable-next=too-many-statements, too-many-branches
def check_dynamic_axes_setting(
    input_signature: list[tuple[str, Union[tuple[int, ...], str]]], dynamic_axes: dict[str, dict[int, dict[str, int]]]
) -> bool:
    """Checks if given dynamic axes setting is valid.

    Args:
        input_signature (list[tuple[str, Union[tuple[int, ...], str]]]): A list of tuples mapping fx graph input names
            to their shape if they are torch.Tensor instances or to their class name otherwise.
        dynamic_axes (dict[str, dict[int, dict[str, int]]]): Dynamic axes setting to examine.

    Returns:
        bool: True if `dynamic_axes` is valid, False otherwise.
    """

    valid = True
    input_signature_dict = dict(input_signature)
    dynamic_dim_size, min_val, opt_val, max_val, test_val = None, None, None, None, None
    for name, axis_setting in dynamic_axes.items():
        signature = input_signature_dict.get(name)
        if signature is None:
            log.error(f"Input '{name}' not found in model input signature")
            valid = False

        if len(axis_setting) == 0:
            log.error(f"Axis to dynamize not given for input '{name}'")
            valid = False

        elif len(axis_setting) > 1:
            log.error(f"Two or more axes given for input '{name}'")
            valid = False

        axis = next(iter(axis_setting))
        if not isinstance(axis, int):
            log.error(f"Non-integer axis given for input '{name}'")
            valid = False

        if not isinstance(signature, (tuple, list)):
            log.error(f"Input '{name}' is not a tensor")
            valid = False
            continue

        if not -len(signature) <= axis < len(signature):
            log.error(f"Invalid axis given for input '{name}'")
            valid = False
            continue

        if dynamic_dim_size and dynamic_dim_size != signature[axis]:
            log.error(
                f"Dimension size({signature[axis]}) of input '{name}' is different from others({dynamic_dim_size})"
            )

        dynamic_dim_size = signature[axis]
        setting = axis_setting.get(axis)
        if setting is None:
            continue

        min_size = setting.get("min")
        opt_size = setting.get("opt")
        max_size = setting.get("max")
        test_size = setting.get("test")

        if min_size is None:
            log.error(f"Dynamic range setting 'min' not given for input '{name}'")
            valid = False

        if opt_size is None:
            log.error(f"Dynamic range setting 'opt' not given for input '{name}'")
            valid = False

        if max_size is None:
            log.error(f"Dynamic range setting 'max' not given for input '{name}'")
            valid = False

        if test_size is None:
            log.error(f"Dynamic range setting 'test' not given for input '{name}'")
            valid = False

        if not (min_size and opt_size and max_size and test_size):
            continue

        if not isinstance(min_size, int):
            log.error(f"Non-integer dynamic range setting 'min' given for input '{name}'")
            valid = False

        if not isinstance(opt_size, int):
            log.error(f"Non-integer dynamic range setting 'opt' given for input '{name}'")
            valid = False

        if not isinstance(max_size, int):
            log.error(f"Non-integer dynamic range setting 'max' given for input '{name}'")
            valid = False

        if not isinstance(test_size, int):
            log.error(f"Non-integer dynamic range setting 'test' given for input '{name}'")
            valid = False

        if min_val and min_val != min_size:
            log.error(f"Dynamic range 'min' setting({min_size}) of input '{name}' is different from others({min_val})")
            valid = False

        if opt_val and opt_val != opt_size:
            log.error(f"Dynamic range 'opt' setting({opt_size}) of input '{name}' is different from others({opt_val})")
            valid = False

        if max_val and max_val != max_size:
            log.error(f"Dynamic range 'max' setting({max_size}) of input '{name}' is different from others({max_val})")
            valid = False

        if test_val and test_val != test_size:
            log.error(
                f"Dynamic range 'test' setting({test_size}) of input '{name}' is different from others({test_val})"
            )
            valid = False

        min_val = min_size
        opt_val = opt_size
        max_val = max_size
        test_val = test_size

        if not min_size <= opt_size <= max_size or not min_size <= test_size <= max_size:
            log.error(f"Invalid dynamic range setting given for input '{name}'")
            valid = False

    return valid


def configure_dynamic_dimensions(
    input_signature: list[tuple[str, Union[tuple[int, ...], str]]], dynamic_axes: dict[str, dict[int, dict[str, int]]]
) -> DynamicDimensions:
    """Configures dynamic dimension setting to be used by `dynamize` with given ONNX proto and dynamic axes setting.

    Args:
        input_signature (list[tuple[str, Union[tuple[int, ...], str]]]): A list of tuples mapping fx graph input names
            to their shape if they are torch.Tensor instances or to their class name otherwise.
        dynamic_axes (Optional[dict[str, dict[int, dict[str, int]]]], optional):
            To specify axes of tensors dynamic(i.e. known only at run-time), set `dynamic_axes` to a dict with schema:

            * KEY (str): an input name.

            * VALUE (dict[int, dict[str, int]]): a single item dictionary whose key is dynamic dimension of input
                and value is a dynamic range setting dictionary containing min, opt, max, test dimension size settings.

    Raises:
        ValueError: When dynamic ONNX proto is given or when invalid `dynamic_axes` is given.

    Returns:
        DynamicDimensions: Dynamic dimension setting to be used as an input of `dynamize`.
    """

    if not check_dynamic_axes_setting(input_signature, dynamic_axes):
        raise ValueError("Invalid dynamic axes setting")

    settings = {}
    dynamic_dim_size = None
    onnx_inputs_dict = dict(input_signature)
    for name, setting in dynamic_axes.items():
        dynamic_axis = next(iter(setting))

        shape = onnx_inputs_dict[name]
        assert shape is not None

        dynamic_dim_size = shape[dynamic_axis]

        min_val = setting[dynamic_axis].get("min")
        max_val = setting[dynamic_axis].get("max")
        opt_val = setting[dynamic_axis].get("opt")
        opt_val = setting[dynamic_axis].get("test")

        if dynamic_axis < 0:
            dynamic_axis = len(shape) + dynamic_axis

        settings[name] = DynamicSetting(shape, dynamic_axis, min_val, max_val, opt_val)  # type: ignore

    assert dynamic_dim_size is not None and isinstance(dynamic_dim_size, int)
    return DynamicDimensions(dynamic_dim_size, settings)


# pylint: disable-next=too-many-locals,too-many-branches, too-many-statements
def dynamize(onnx_proto: ModelProto, dynamic_dims: DynamicDimensions) -> ModelProto:
    """Dynamizes given ONNX proto with given dynamic dimension setting.

    Args:
        onnx_proto (ModelProto): ONNX model proto to dynamize.
        dynamic_dims (DynamicDimensions): Dynamic dimension setting configured by `configure_dynamic_dimensions`.

    Raises:
        ValueError: When dynamic ONNX proto is given.
        NotImplementedError: When dynamizing ONNX proto with reshapes with dynamic target shape is attempted.
        RuntimeError: When attempt to dynamize given ONNX proto has failed.

    Returns:
        ModelProto: Dynamized ONNX proto.
    """

    if any(
        any(not isinstance(s, int) or s == -1 for s in extract_tensor_shape(vi) or [])
        for vi in onnx_proto.graph.value_info
    ):
        raise ValueError("Dynamic ONNX proto given")

    input_shapes = [extract_tensor_shape(input) for input in onnx_proto.graph.input]
    if not all(shape and -1 not in shape and all(isinstance(s, int) for s in shape) for shape in input_shapes):
        raise ValueError("Dynamic ONNX proto given")

    graph = gs.import_onnx(onnx_proto)

    for node in graph.nodes:
        if node.op == "Reshape":
            if isinstance(node.inputs[1], gs.Variable):
                raise NotImplementedError("Dynamizing reshapes with dynamic target shape is not supported yet")

            if -1 in node.inputs[1].values:
                node.inputs[1].values = np.array(node.outputs[0].shape)

    dfs_stack: list[tuple[int, Sequence[int], gs.Node]] = []
    visited: set[str] = set()
    handled_tensors: set[str] = set()
    for input_tensor in graph.inputs:
        setting = dynamic_dims.get(input_tensor.name)
        if setting is None:
            continue

        dynamic_dim = setting.dim
        # pylint: disable-next=cell-var-from-loop
        dfs_stack.extend(
            [
                (
                    dynamic_dim,
                    input_tensor.shape,
                    node,
                )
                for node in input_tensor.outputs
            ]
        )
        new_shape = list(input_tensor.shape[:])
        new_shape[dynamic_dim] = "N"
        input_tensor.shape = new_shape

    while len(dfs_stack) > 0:
        input_dynamic_dim, input_shape, node = dfs_stack.pop()

        if node.name in visited:
            continue

        output_dynamic_dim = input_dynamic_dim

        if node.op == "Reshape":
            if node.inputs[1].name in handled_tensors:
                continue

            target_shape = node.inputs[1].values

            elements_til_batch = 0
            for i in range(input_dynamic_dim + 1):
                elements_til_batch = (elements_til_batch if elements_til_batch else 1) * input_shape[i]

            acc = 1
            for i, dim in enumerate(target_shape):
                if dim == dynamic_dims.dim_size:
                    output_dynamic_dim = i
                    break

                acc = acc * dim
                if acc >= elements_til_batch:
                    output_dynamic_dim = i
                    break

            target_shape[output_dynamic_dim] = -1
            handled_tensors.add(node.inputs[1].name)

        elif node.op == "Resize":
            axes = node.attrs.get("axes")
            if axes is None or len(axes) == len(node.outputs[0].shape):
                if node.inputs[3].name != "":  # resize with shape
                    assert node.inputs[2].name == ""

                    if isinstance(node.inputs[3], gs.Variable):
                        raise ValueError("Dynamic ONNX proto given")

                    target_shape = node.inputs[3].values

                    if target_shape[input_dynamic_dim] != node.outputs[0].shape[input_dynamic_dim]:
                        raise ValueError("Dynamic ONNX proto given")

                    node.inputs[2] = gs.Constant(
                        f"{node.name}_scale", np.array(target_shape / node.inputs[0].shape, dtype=np.float32)
                    )
                    node.inputs[3] = gs.Variable.empty()

            elif input_dynamic_dim in axes:
                raise ValueError("Dynamic ONNX proto given")

        elif node.op == "Transpose":
            target_permutation = node.attrs.get("perm")
            output_dynamic_dim = target_permutation.index(input_dynamic_dim)

        for output in node.outputs:
            # pylint: disable-next=cell-var-from-loop
            children = [(output_dynamic_dim, output.shape[:], child) for child in output.outputs]

            if children:
                # dfs_stack.extend(reversed(children))
                dfs_stack.extend(children)

        visited.add(node.name)

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

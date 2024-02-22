import inspect
from typing import Any, Callable, Optional, Union

import onnx_graphsurgeon as gs
import torch
from onnx import ModelProto
from torch.fx.graph_module import GraphModule

from ...options import DynamicAxisOptions, DynamicInputOptions


class DynamicSignature(list[tuple[str, tuple[Union[int, str, tuple[int, ...]], ...]]]):
    """The input signature of a model which have dynamic input shape"""


class Signature(list[tuple[str, tuple[int, ...]]]):
    """The input signature of a model"""

    @property
    def is_dynamic(self) -> bool:
        """Whether this signature contains any dynamic shape or not."""
        return any((-1 in shape) for _, shape in self)

    def asdict(self) -> dict[str, tuple[int, ...]]:
        """Reinterpret this signature as a dictionary whose keys are the names of each input tensor
        mapped to the shape of the input tensor"""
        return dict(*self)

    def get(self, name: str) -> Optional[tuple[int, ...]]:
        """Gets the shape of the input tensor named `name`

        Args:
            name (str): The name of an input tensor.

        Returns:
            Optional[tuple[int, ...]]: The shape of the input tensor if found, `None` otherwise.
        """
        return self.asdict().get(name)

    @classmethod
    def from_onnx(
        cls, proto_or_graph: Union[ModelProto, gs.Graph], options: Optional[DynamicAxisOptions] = None
    ) -> Union["Signature", DynamicSignature]:
        """Creates the signature from an ONNX proto or an ONNX graph.

        Args:
            proto_or_graph (Union[ModelProto, gs.Graph]): An ONNX proto or an ONNX graph.
            options (Optional[DynamicAxisOptions]): Optional dynamic input options. Defaults to `None`.

        Returns:
            Union["Signature", DynamicSignature]: A `Signature` object if `options` is `None`,
                `DynamicSignature` object otherwise.
        """
        graph = gs.import_onnx(proto_or_graph) if isinstance(proto_or_graph, ModelProto) else proto_or_graph
        signature = cls(
            (input_tensor.name, tuple((-1 if isinstance(s, str) else s) for s in input_tensor.shape))
            for input_tensor in graph.inputs
            if isinstance(input_tensor, gs.Variable) and input_tensor.shape is not None
        )
        if options is not None:
            return dynamize_signature(signature, options)
        return signature

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        options: Optional[DynamicAxisOptions] = None,
    ) -> Union["Signature", DynamicSignature]:
        """Creates the signature from a module and inputs provided for its forward method.

        Args:
            module (torch.nn.Module): A module.
            args (tuple[Any, ...]): Arguments to be provided for the module's forward method.
            kwargs (dict[str, Any]): Keyword arguments to be provided for the module's forward method.
            options (Optional[DynamicAxisOptions]): Optional dynamic input options. Defaults to `None`.

        Returns:
            Union["Signature", DynamicSignature]: A `Signature` object if `options` is `None`,
                `DynamicSignature` object otherwise.
        """
        if isinstance(module, GraphModule) and isinstance(original_params := module.meta.get("original_params"), dict):
            modified_params = inspect.signature(module.forward).parameters
            signature_map = {
                k: v for k, v in map_signature(original_params, *args, **kwargs).items() if k in modified_params
            }
        else:
            signature_map = map_signature(module.forward, *args, **kwargs)
        signature = cls(
            (
                name,
                tuple(value.shape) if isinstance(value, torch.Tensor) else (),
            )
            for name, value in signature_map.items()
        )
        if options is not None:
            return dynamize_signature(signature, options)
        return signature


def dynamize_signature(input_signature: Signature, options: DynamicAxisOptions) -> DynamicSignature:
    """Creates dynamic signature from a static input signature using the dynamic input options.

    Args:
        input_signature (Signature): An input signature.
        options (DynamicAxisOptions): A dynamic export options.

    Returns:
        DynamicSignature: The converted dynamic signature.
    """
    new_input_signature: DynamicSignature = DynamicSignature()
    for name, shape in input_signature:
        axis_options = options.get(name)
        if axis_options is None:
            continue
        axis = axis_options.axis
        new_input_signature.append((name, tuple("N" if i == axis else s for i, s in enumerate(shape))))
    return new_input_signature


def update_dynamic_signature(input_signature: DynamicSignature, options: DynamicInputOptions) -> DynamicSignature:
    """Updates signature with dynamic input options for TensorRT benchmark

    Args:
        input_signature (DynamicSignature): current input signature with dynamic shape
        options (DynamicInputOptions): dynamic input sizes for benchmarking

    Returns:
        DynamicSignature: input signature with dynamic input options
    """
    new_input_signature: DynamicSignature = DynamicSignature([])
    for name, shape in input_signature:
        size_options = options.get(name)
        if size_options is None:
            continue
        range_setting = (
            size_options.min,
            size_options.opt,
            size_options.max,
            size_options.test,
        )
        new_input_signature.append((name, tuple(s if isinstance(s, int) else range_setting for s in shape)))
    return new_input_signature


# pylint: disable=protected-access
def map_signature(
    func_or_its_params: Union[Callable, dict[str, inspect.Parameter]], *args: Any, **kwargs: Any
) -> dict[str, Any]:
    """Maps the parameter names of a function to the corresponding values passed in args and kwargs.

    This function returns a list of tuples, where each tuple contains a parameter name and its corresponding value.
    If a parameter name exists in the kwargs dictionary, its value is taken from there. Otherwise, the values are taken
    in order from the args tuple. If there are no values left in args or kwargs, the default value of the parameter
    (if it exists) is used.

    Args:
        func_or_its_params (Union[Callable, dict[str, inspect.Parameter]]): Function to inspect or its parameters
        args (Any): Positional arguments.
        kwargs (Any): Keyword arguments.

    Returns:
        dict[str, Any]: List of tuples mapping parameter names to their values.

    Note:
        This function assumes that `args` and `kwargs` match the exact function signature,
        in order and length. If they don't, the result may not be as expected or exceptions might occur.
    """
    params = (
        dict(inspect.signature(func_or_its_params).parameters.items())
        if callable(func_or_its_params)
        else func_or_its_params
    )

    var_pos: Optional[tuple[int, str]] = None
    var_key: Optional[tuple[int, str]] = None
    # mapped: dict[str, Any] = {name: inspect._empty for name in params}
    mapped: dict[str, Any] = {}
    for i, (name, param) in enumerate(params.items()):
        if param.kind == inspect._ParameterKind.VAR_POSITIONAL:
            var_pos = (i, name)
            mapped[name] = ()
        if param.kind == inspect._ParameterKind.VAR_KEYWORD:
            var_key = (i, name)
            mapped[name] = {}

    for name, val in kwargs.items():
        if name in params:
            mapped[name] = val
            params.pop(name)
            continue
        if var_key is not None:
            var_key_name = var_key[1]
            mapped[var_key_name][name] = val
            params.pop(var_key_name)
            continue

    names = list(params)
    for i, val in enumerate(args):
        if i < len(names):
            name = names[i]
            mapped[name] = val
            continue
        if var_pos is not None and i >= var_pos[0]:
            var_pos_name = var_pos[1]
            mapped[var_pos_name] += (val,)

    return mapped

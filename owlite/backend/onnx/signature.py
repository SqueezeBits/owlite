import inspect
from typing import Any, Callable, Optional, Union

import onnx_graphsurgeon as gs
import torch
from onnx import ModelProto

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
        signature_map = map_signature(module.forward, *args, **kwargs)
        signature = cls(
            (
                name,
                tuple(value.shape) if isinstance(value, torch.Tensor) else (),
            )
            for name, value in signature_map
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

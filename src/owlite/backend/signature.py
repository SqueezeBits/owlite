import inspect
import json
from copy import deepcopy
from typing import Any, Callable, Optional, Union

import onnx_graphsurgeon as gs
import torch
from onnx import ModelProto
from torch.fx.graph_module import GraphModule

from ..options import DynamicAxisOptions, DynamicInputOptions
from ..owlite_core.logger import log
from .utils import normalize_parameter_name


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

    def __str__(self) -> str:
        return json.dumps(self)

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
        if isinstance(module, GraphModule):
            signature_map = map_graph_module_signature(module, *args, **kwargs)
        else:
            signature_map = map_signature(module.forward, *args, **kwargs)

        signature = cls()
        for name, value in signature_map.items():
            if isinstance(value, torch.Tensor):
                signature.append((name, tuple(value.shape)))
                continue
            if isinstance(value, tuple):
                signature.extend((f"{name}_{i}", t.shape) for i, t in enumerate(value) if isinstance(t, torch.Tensor))
                continue
            if isinstance(value, dict):
                signature.extend((k, t.shape) for k, t in value.items() if isinstance(t, torch.Tensor))
                continue
            signature.append((name, ()))

        if options is not None:
            return dynamize_signature(signature, options)
        return signature

    @classmethod
    def from_str(
        cls,
        string: str,
    ) -> Union["Signature", DynamicSignature]:
        """Creates the signature from string provided for API response.
        Args:
            string (str): A string.
        Returns:
            Union["Signature", DynamicSignature]: A `Signature` object if `options` is `None`,
                `DynamicSignature` object otherwise.
        """
        signature_list = json.loads(string)
        signature = cls((name, tuple(value)) for name, value in signature_list if isinstance(name, str))
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
        else deepcopy(func_or_its_params)
    )

    names = list(params)
    var_pos: Optional[tuple[int, str]] = None
    var_key: Optional[tuple[int, str]] = None
    mapped: dict[str, Any] = {}

    for i, (name, param) in enumerate(params.items()):
        if param.kind == inspect._ParameterKind.VAR_POSITIONAL:
            var_pos = (i, name)
            mapped[name] = ()
        if param.kind == inspect._ParameterKind.VAR_KEYWORD:
            var_key = (i, name)
            mapped[name] = {}

    for i, val in enumerate(args):
        if var_pos is not None and i >= var_pos[0]:
            mapped[var_pos[1]] += (val,)
            continue
        mapped[names[i]] = val

    for name, val in kwargs.items():
        if var_key is not None and name not in names:
            mapped[var_key[1]][name] = val
            continue
        mapped[name] = val

    return mapped


def map_graph_module_signature(
    module: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Maps the args and kwargs to the parameters of the forward method of a graph module
    generated by `owlite.fx.symbolic_trace`. If the graph module doesn't have meta data 'original_params',
    automatically falls back to `map_signature`.

    Args:
        module (GraphModule): a graph module generated by `owlite.fx.symbolic_trace`
        args (Any): Positional arguments.
        kwargs (Any): Keyword arguments.

    Returns:
        dict[str, Any]: the mapped signatures
    """
    if (original_params := module.meta.get("original_params", None)) is None:
        log.debug_warning("This graph module has no meta data 'original_params'")
        return map_signature(module.forward, *args, **kwargs)
    modified_params = {normalize_parameter_name(k): v for k, v in inspect.signature(module.forward).parameters.items()}
    mapped_signature = map_signature(original_params, *args, **kwargs)
    signature_map: dict[str, Any] = {}
    for p, (k, v) in enumerate(mapped_signature.items()):
        if k in modified_params:
            signature_map[k] = v
            continue
        if isinstance(v, tuple):
            # variadic positional arguments are flattened by torch.compile
            # e.g. `def forward(self, *args, x)` -> `def forward(self, args_0, args_1, x)`
            # or `def forward(self, args_0_, args_1_, x)`
            # when two arguments are provided by the user, depending on torch version and the host OS
            success = True
            for i, x in enumerate(v):
                for name in (f"{k}_{i}", f"{k}_{i}_"):
                    if name in modified_params:
                        signature_map[name] = x
                        break
                else:
                    success = False
                    log.debug_warning(
                        f"Failed to map {i}-th variadic positional argument {k} of the graph module's forward method"
                    )
            if success:
                continue
        if isinstance(v, dict):
            # variadic keyword arguments are flattened by torch.compile
            # e.g. `def forward(self, x, **kwargs)` -> `def forward(self, x, y, z)`
            # when the model was called as `output = model(a, y=b, z=c)`
            for name, x in v.items():
                if name in modified_params:
                    signature_map[name] = x
                else:
                    log.debug_warning(
                        f"Failed to map the variadic positional argument {k} with key {name} "
                        "of the graph module's forward method"
                    )
            continue
        if any(arg_name in modified_params for arg_name in (f"args_{p}", f"args_{p}_")):
            # Rarely, arguments can be squashed as a variadic position argument `args`
            signature_map[k] = v
            continue
        log.debug_warning(f"Failed to map signature of {p}-th parameter {k} of the graph module's forward method")
    return signature_map
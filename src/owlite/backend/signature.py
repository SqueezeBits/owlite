import inspect
import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import onnx_graphsurgeon as gs
import torch
from onnx import ModelProto
from torch.fx.graph_module import GraphModule
from typing_extensions import Self

from ..options import DynamicAxisOptions, DynamicInputOptions
from ..owlite_core.logger import log
from .utils import normalize_parameter_name


class Signature(dict[str, list[int | str | list[int]]]):
    """The input signature of a model."""

    def __str__(self) -> str:
        return self.dumps()

    @property
    def is_dynamic(self) -> bool:
        """Check whether this signature contains any dynamic shape or not."""
        return any(("N" in shape) or any(isinstance(s, list) for s in shape) for _, shape in self.items())

    @classmethod
    def from_onnx(cls, proto_or_graph: ModelProto | gs.Graph, options: DynamicAxisOptions | None = None) -> Self:
        """Create the signature from an ONNX proto or an ONNX graph.

        Args:
            proto_or_graph (ModelProto | gs.Graph): An ONNX proto or an ONNX graph.
            options (DynamicAxisOptions | None): Optional dynamic input options. Defaults to `None`.

        Returns:
            Signature: The created `Signature` object.
        """
        graph = gs.import_onnx(proto_or_graph) if isinstance(proto_or_graph, ModelProto) else proto_or_graph
        signature = cls(
            (input_tensor.name, ["N" if isinstance(s, str) else s for s in input_tensor.shape])
            for input_tensor in graph.inputs
            if isinstance(input_tensor, gs.Variable) and input_tensor.shape is not None
        )
        if options is not None:
            signature.mark_dynamic_axes(options)

        return signature

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        options: DynamicAxisOptions | None = None,
    ) -> Self:
        """Create the signature from a module and inputs provided for its forward method.

        Args:
            module (torch.nn.Module): A module.
            args (tuple[Any, ...]): Arguments to be provided for the module's forward method.
            kwargs (dict[str, Any]): Keyword arguments to be provided for the module's forward method.
            options (DynamicAxisOptions | None): Optional dynamic input options. Defaults to `None`.

        Returns:
            Signature: The created `Signature` object.
        """
        if isinstance(module, GraphModule):
            signature_map = map_graph_module_signature(module, *args, **kwargs)
        else:
            signature_map = map_signature(module.forward, *args, **kwargs)

        signature = cls()
        for name, value in signature_map.items():
            if isinstance(value, torch.Tensor):
                signature[name] = list(value.shape)
                continue
            if isinstance(value, tuple):
                signature.update(
                    {f"{name}_{i}": list(t.shape) for i, t in enumerate(value) if isinstance(t, torch.Tensor)}
                )
                continue
            if isinstance(value, dict):
                signature.update({k: list(t.shape) for k, t in value.items() if isinstance(t, torch.Tensor)})
                continue
            signature[name] = []

        if options is not None:
            signature.mark_dynamic_axes(options)

        return signature

    @classmethod
    def from_str(
        cls,
        string: str,
    ) -> Self:
        """Create the signature from string provided by API response.

        Args:
            string (str): JSON dumped string.

        Returns:
            Signature: The created `Signature` object.
        """
        # Cast the return value of json.loads to a dictionary for backwards compatibility.
        return cls(dict(json.loads(string)))

    def dumps(self) -> str:
        """Dump signature to string.

        Returns:
            str: The dumped string.
        """
        # Currently, shape checking logic in BE cannot properly handle dictionary dumped strings
        # so, dump in legacy format(list version) for now. This should be removed when BE becomes
        # capable of handling dictionaries.
        return json.dumps(list(self.items()))

    def mark_dynamic_axes(self, options: DynamicAxisOptions) -> None:
        """Mark dynamic axes of the signature with placeholder value.

        Args:
            options (DynamicAxisOptions): The dynamic axis options to apply.
        """
        check_names_in_options(self, options)

        for name, axis in options.items():
            sig = self.get(name, [])
            try:
                sig[axis] = "N"
            except IndexError as e:
                log.error(f"Axis {axis} is not valid for tensor with rank {len(sig)}")
                raise e

    def fill_dynamic_ranges(self, options: DynamicInputOptions) -> None:
        """Fill dynamic range settings for the benchmark.

        Args:
            options (DynamicInputOptions): The dynamic input options to apply.
        """
        check_names_in_options(self, options)

        for name, option in options.items():
            if (sig := self.get(name, None)) is None:
                raise ValueError("Invalid tensor name")

            try:
                dynamic_axis = sig.index("N")
            except ValueError as e:
                raise ValueError(f"Tensor({name}) is not marked as dynamic") from e

            dynamic_range = option.to_list()
            sig[dynamic_axis] = dynamic_range


def check_names_in_options(input_signature: Signature, options: DynamicAxisOptions | DynamicInputOptions) -> None:
    """Check if tensor names in options are valid.

    Args:
        input_signature (Signature): The input signature to apply options to.
        options (DynamicAxisOptions | DynamicInputOptions): The option to apply.

    Raises:
        ValueError: When tensor names in options are not found in input signature.
    """
    names_in_signature = set(input_signature.keys())
    names_in_options = set(options.keys())
    invalid_names = names_in_options.difference(names_in_signature)
    if invalid_names:
        raise ValueError(f"Invalid tensor name: {invalid_names}")


# pylint: disable=protected-access
def map_signature(
    func_or_its_params: Callable | dict[str, inspect.Parameter], *args: Any, **kwargs: Any
) -> dict[str, Any]:
    """Map the parameter names of a function to the corresponding values passed in args and kwargs.

    This function returns a list of tuples, where each tuple contains a parameter name and its corresponding value.
    If a parameter name exists in the kwargs dictionary, its value is taken from there. Otherwise, the values are taken
    in order from the args tuple. If there are no values left in args or kwargs, the default value of the parameter
    (if it exists) is used.

    Args:
        func_or_its_params (Callable | dict[str, inspect.Parameter]): Function to inspect or its parameters
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
    var_pos: tuple[int, str] | None = None
    var_key: tuple[int, str] | None = None
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
    """Map arguments and keyword arguments to the parameters of a graph module's forward method.

    This function maps the provided args and kwargs to the parameters of the forward method of a graph module
    generated by owlite.fx.symbolic_trace. If the graph module does not have metadata named 'original_params',
    it automatically falls back to using map_signature.


    Args:
        module (GraphModule): a graph module
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

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

from ..enums import ModelStatus
from ..options import DynamicAxisOptions, DynamicInputOptions
from ..owlite_core.logger import log


# TODO(huijong): replace this class with inspect.Signature
class Signature(dict[str, list[int | str | list[int]]]):
    """The input signature of a model."""

    def __init__(self, *args: Any, unused_param_names: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.unused_param_names = unused_param_names or []

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

        Parameters that are not used in given module's graph will not appear in the mapped signature if given
        module is an instance of `torch.nn.GraphModule`.

        Args:
            module (torch.nn.Module): A module.
            args (tuple[Any, ...]): Arguments to be provided for the module's forward method.
            kwargs (dict[str, Any]): Keyword arguments to be provided for the module's forward method.
            options (DynamicAxisOptions | None): Optional dynamic input options. Defaults to `None`.

        Returns:
            Signature: The created `Signature` object.
        """
        # This lazy import is required for avoiding circular import error
        # pylint: disable-next=import-outside-toplevel
        from .fx.node import find_placeholders

        signature_map = map_signature(module.forward, *args, **kwargs)

        unused_param_names = []
        if isinstance(module, GraphModule) and (module.meta.get("status", None) == ModelStatus.TRACED):
            keys = list(signature_map.keys())
            placeholders = find_placeholders(module.graph)
            unused_param_names = [
                sig_key for sig_key, placeholder_node in zip(keys, placeholders) if not placeholder_node.users
            ]
            signature_map = {k: v for k, v in signature_map.items() if k not in unused_param_names}

        signature = cls(unused_param_names=unused_param_names)
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

    def warn_signature_change(self, original_params_dict: dict[str, inspect.Parameter]) -> None:
        """Warn users for any differences compared to the given original parameters.

        Args:
            original_params_dict (dict[str, inspect.Parameter]): a parameter dictionary to compare to.
        """
        if len(self) == len(original_params_dict):
            return

        indent4 = " " * 4
        indent8 = " " * 8
        unused_but_required_flag = False

        def red(s: str) -> str:
            return f"\033[91m{s}\033[0m"

        def yellow(s: str) -> str:
            return f"\u001b[33m{s}\033[0m"

        def strikethrough(s: str) -> str:
            return "\u0336".join((*s, ""))

        def as_snippet(param: inspect.Parameter, is_last: bool) -> str:
            nonlocal unused_but_required_flag

            default_exists = param.default is not inspect.Parameter.empty

            snippet = f"{param.name}={param.default}" if default_exists else param.name
            comment = ""
            if param.name not in self and param.name not in self.unused_param_names:  # removed
                snippet = red(strikethrough(snippet))
                comment = "is removed"

            elif param.name in self.unused_param_names:  # kept but unused
                snippet = yellow(snippet)
                epilog = "and optional" if default_exists else f"but {red('required')}"
                unused_but_required_flag = unused_but_required_flag or not default_exists
                comment = f"is unused {epilog}"

            snippet = f"{indent8}{snippet}{'' if is_last else ','}"
            if comment:
                snippet += f"  # <--- {comment}"
            return snippet

        params_snippet = "\n".join(
            as_snippet(param, i == len(original_params_dict) - 1)
            for i, param in enumerate(original_params_dict.values())
        )
        code_snippet = f"{indent4}def forward(\n{indent8}self,\n{params_snippet}\n{indent4}):\n{indent8}..."
        extra_instruction = ""

        if unused_but_required_flag:
            extra_instruction = (
                "However, you might still need to provide some of the unused parameters to the model "
                "in order to synchronize the input signature with the original model. "
            )
        log.warning(
            "The model has unused parameters (i.e. inputs that does not affect model's output(s)). "
            f"{extra_instruction}"
            "See the comments in the following code snippet for more details:\n"
            f"{code_snippet}"
        )

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

# pylint: disable=protected-access, too-many-statements
import inspect
import logging
import sys
import traceback
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Optional, Union

import torch
import torch._dynamo as torch_dynamo
from torch import Tensor
from torch.fx import Node
from torch.fx.graph_module import GraphModule, _WrappedCall
from torch.nn.parallel import DataParallel, DistributedDataParallel

from ...enums import OwLiteStatus, ParamStatus
from ...owlite_core.logger import log
from ..config import FORCE_OUTPUT_COMPATIBILITY
from ..signature import map_signature
from ..utils import (
    get_most_common_device,
    get_most_common_floating_point_type,
    move_tensors_to,
)
from .node import find_the_output_node
from .transforms import apply_graph_module_transforms


# pylint: disable-next=missing-class-docstring, too-few-public-methods
class BackendProvider:
    def __init__(self) -> None:
        self.graph_module: Optional[GraphModule] = None

    def __call__(self, graph_module: GraphModule, inputs: Sequence[Tensor]) -> Callable[..., Tensor]:
        self.graph_module = graph_module
        return graph_module.forward


# pylint: disable-next=missing-class-docstring, too-few-public-methods
class TensorPlaceholder:
    pass


def extract_structure(nested_tensor: Any) -> Any:
    """Extracts nested structure of given nested tensor.

    Args:
        nested_tensor (Any): Tensor nested in arbitrary data structure.

    Raises:
        ValueError: When `nested_tensor` contains any other objects than
            `Tensor`, `tuple`, `list`, `OrderedDict` and `dict`.

    Returns:
        Any: The `nested_tensor` but, occurrences of `Tensor` replaced with `TensorPlaceHolder`.
    """

    def _extract_structure(obj: Any) -> Any:
        if isinstance(obj, Tensor):
            return TensorPlaceholder()
        if isinstance(obj, tuple):
            return tuple(_extract_structure(e) for e in obj)
        if isinstance(obj, list):
            return [_extract_structure(e) for e in obj]
        if isinstance(obj, OrderedDict):
            return OrderedDict({key: _extract_structure(val) for (key, val) in obj.items()})
        if isinstance(obj, dict):
            return {key: _extract_structure(val) for (key, val) in obj.items()}

        raise TypeError(f"Cannot extract structure from an object of type: {type(obj)}")

    return _extract_structure(nested_tensor)


def flatten_output(output: Union[Iterable, Tensor, Node]) -> list[Union[Tensor, Node]]:
    """Flattens the output of `nn.Module` or the argument of fx output node.

    Args:
        output (Union[Iterable, Tensor, Node]): The output to flatten.

    Returns:
        list[Union[Tensor, Node]]: The flattened output.

    Notes:
        This function assumes all nodes passed as an argument to fx output node produces
        `Tensor` and the orderings of the outputs are preserved during `torch.compile`.
    """

    flattened_output: list[Union[Tensor, Node]] = []

    def _extract_output(output: Union[Iterable, Tensor, Node]) -> None:
        if isinstance(output, Tensor):
            flattened_output.append(output)
        elif isinstance(output, Node):
            flattened_output.append(output)
        elif isinstance(output, (dict, OrderedDict)):
            for out in output.values():
                _extract_output(out)
        elif isinstance(output, Iterable):
            for out in output:
                _extract_output(out)

    _extract_output(output)

    return flattened_output


def filter_output(graph_module_output: Any, original_output: Any) -> list[Node]:
    """Filters `graph_module_output`.

    Args:
        graph_module_output (Any): The argument of fx output node to filter.
        original_output (Any): The output of an original model before `torch.compile`.

    Returns:
        Any: The filtered and flattened output.
    """

    flattened_graph_module_output = flatten_output(graph_module_output)
    flattened_original_output = flatten_output(original_output)

    num_graph_module_output = len(flattened_graph_module_output)
    num_original_output = len(flattened_original_output)

    assert num_graph_module_output >= num_original_output

    return flattened_graph_module_output[: len(flattened_original_output)]  # type: ignore


# pylint: disable-next=missing-function-docstring
def insert_output_adapter(graph_module: GraphModule, original_output: Any) -> GraphModule:
    original_output_structure = extract_structure(original_output)

    graph = graph_module.graph
    output_node = find_the_output_node(graph)
    filtered_graph_module_output = filter_output(output_node.args, original_output)

    with graph.inserting_before(output_node):
        output_adapter_func = create_output_adapter(original_output_structure)
        adapter_node = graph.call_function(output_adapter_func, (filtered_graph_module_output,))
        graph.output(adapter_node)

        graph.erase_node(output_node)

        graph.lint()
        graph.eliminate_dead_code()
        graph_module.recompile()

    return graph_module


# pylint: disable-next=missing-function-docstring
def create_output_adapter(output_structure: Any) -> Callable[[Any], Any]:
    def output_adapter(graph_module_out: list[Tensor]) -> Any:
        # output adapter node will be transparent to torch.onnx.export(jit.trace)
        if torch._C._get_tracing_state():  # pylint: disable=protected-access
            return graph_module_out

        # rearrange output to match with original structure
        def _create_return(obj: Any) -> Any:
            if isinstance(obj, TensorPlaceholder):
                return graph_module_out.pop(0)
            if isinstance(obj, tuple):
                return tuple(_create_return(e) for e in obj)
            if isinstance(obj, list):
                return [_create_return(e) for e in obj]
            if isinstance(obj, OrderedDict):
                return OrderedDict({key: _create_return(val) for (key, val) in obj.items()})
            if isinstance(obj, dict):
                return {key: _create_return(val) for (key, val) in obj.items()}

            return tuple(graph_module_out)

        return _create_return(output_structure)

    return output_adapter


# pylint: disable-next=missing-function-docstring
def patched_wrapped_call(self: Any, obj: GraphModule, *_args: Any, **_kwargs: Any) -> Any:
    param_status: dict[str, ParamStatus] = obj.meta.get("param_status", None)

    if param_status:
        ignored_keys = [name for name, status in param_status.items() if status != ParamStatus.ALIVE]
        kwargs = {key: arg for key, arg in _kwargs.items() if key not in ignored_keys}

        kept_names = [name for name, status in param_status.items() if status != ParamStatus.PURGED]
        args = tuple(
            a
            for i, a in enumerate(_args)
            if i < len(kept_names) and param_status.get(kept_names[i]) == ParamStatus.ALIVE
        )
        if log.level <= logging.DEBUG:
            log.debug(
                f"{len(_kwargs) - len(kwargs)} out of {len(_kwargs)} kwargs dropped. "
                f"({[key for key in ignored_keys if key in _kwargs]})"
            )
            dropped_indices = tuple(
                i for i in range(len(_args)) if i < len(kept_names) and param_status[kept_names[i]] != ParamStatus.ALIVE
            )
            log.debug(f"{len(_args) - len(args)} out of {len(_args)} args dropped. ({dropped_indices})")
    else:
        args = _args
        kwargs = _kwargs

    # pylint: disable-next=broad-exception-caught
    # ruff: noqa: B904
    try:
        if self.cls_call is not None:
            return self.cls_call(obj, *args, **kwargs)
        return super(self.cls, obj).__call__(*args, **kwargs)
    except Exception as e:
        assert e.__traceback__
        topmost_framesummary: traceback.FrameSummary = traceback.StackSummary.extract(
            traceback.walk_tb(e.__traceback__)
        )[-1]
        if "eval_with_key" in topmost_framesummary.filename:
            print(
                _WrappedCall._generate_error_message(topmost_framesummary),
                file=sys.stderr,
            )
            raise e.with_traceback(None)
        raise e


_WrappedCall.__call__ = patched_wrapped_call  # type: ignore[method-assign]


def symbolic_trace(model: torch.nn.Module, *args: Any, **kwargs: Any) -> GraphModule:
    """Like `torch.fx.symbolic_trace`, this function traces the input `model` to convert it into a GraphModule.
    In order for the tracing to be successful, the `model` must be able to pass `torch.compile(model, fullgraph=True)`.

    Args:
        model (torch.nn.Module): a torch.nn.Module instance.

    Raises:
        TypeError: if the `model` is not an instance of `torch.nn.Module`
        RuntimeError: if the tracing fails.

    Returns:
        GraphModule: the converted GraphModule.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module instance but object of type {type(model)} given: {model}")
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        _model_type = f"torch.nn.parallel.{type(model).__name__}"
        log.error(
            f"{_model_type} is not supported by symbolic trace, please use 'attribute' module to unwrap model "
            f"from {_model_type}. Try owlite.fx.symbolic_trace(model.module, ...)"
        )
        raise TypeError(f"{_model_type} is not supported by symbolic trace")
    training_status = model.training
    # move input args and kwargs to model device
    device = get_most_common_device(model)
    dtype = get_most_common_floating_point_type(model)
    log.debug(f"Tracing with device={device}, dtype={dtype}")

    args = move_tensors_to(args, device, dtype)
    kwargs = move_tensors_to(kwargs, device, dtype)

    backend = BackendProvider()
    torch_dynamo.reset()
    optimized_model = torch.compile(model, fullgraph=True, backend=backend)
    output = optimized_model(*args, **kwargs)

    graph_module = backend.graph_module

    if graph_module is None:
        raise RuntimeError("Failed to create torch.fx.GraphModule while running optimized model")

    try:
        graph_module = insert_output_adapter(graph_module, output)
    except TypeError as e:
        if FORCE_OUTPUT_COMPATIBILITY:
            log.error(
                "Currently, OwLite can handle models whose output is a (possibly nested) `list`, `tuple`, `dict` or "
                "`OrderedDict` of `Tensor` objects. You can ignore this error by setting the environment variable "
                "OWLITE_FORCE_OUTPUT_COMPATIBILITY to 0. You can also overwrite the value by adding "
                "`owlite.config.FORCE_OUTPUT_COMPATIBILITY=False` before conversion. However, by doing so, your "
                "model's output can lose its nested structure and will be (usually) converted as a flattened `tuple` "
                "of `Tensor` objects."
            )
            raise RuntimeError from e

        log.warning(
            "The converted model's output might lose its nested structure. (e.g. if your original model's output "
            "was conformed with `tuple[Tensor, tuple[Tensor, Tensor]]`, the converted model's output will be "
            "flattened as `tuple[Tensor, Tensor, Tensor]`.) This means that you might need to change your "
            "training or inference code to use the converted model."
        )

    original_params = {**inspect.signature(model.forward).parameters}
    graph_module.meta["original_params"] = original_params

    graph_module = apply_graph_module_transforms(graph_module, args, kwargs)
    graph_module.train(training_status)
    graph_module.meta["owlite_status"] = OwLiteStatus.NOT_COMPRESSED

    graph_module_params = {**inspect.signature(graph_module.forward).parameters}
    log.debug(f"original_params: {[*original_params.keys()]}")
    log.debug(f"graph_module_params: {[*graph_module_params.keys()]}")

    if any(
        param.kind in (inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD)
        for param in original_params.values()
    ):
        log.debug_warning("The original module has variadic positional or keyword argument")
        return graph_module

    if not all(name in original_params for name in graph_module_params):
        log.debug_warning("torch.compile has completely renamed all parameters")
        return graph_module

    received_values = map_signature(model.forward, *args, **kwargs)
    if log.level <= logging.DEBUG:
        for name, val in received_values.items():
            log.debug(f"{name}: {'empty' if val is inspect._empty else 'received'}")

    param_status = {
        name: (
            ParamStatus.ALIVE
            if name in graph_module_params
            else (ParamStatus.KEPT if name in received_values else ParamStatus.PURGED)
        )
        for name in original_params
    }
    if log.level <= logging.DEBUG:
        for name, status in param_status.items():
            log.debug(f"{name}: {status.name}")

    graph_module.meta["param_status"] = param_status

    if not param_status:
        return graph_module

    def warn_dropped_inputs() -> None:
        indent4 = " " * 4
        indent8 = " " * 8

        def red(s: str) -> str:
            return f"\033[91m{s}\033[0m"

        def yellow(s: str) -> str:
            return f"\u001b[33m{s}\033[0m"

        def strikethrough(s: str) -> str:
            return "\u0336".join((*s, ""))

        def as_snippet(param: inspect.Parameter, status: ParamStatus) -> str:
            default_exists = has_default(param)
            snippet = f"{param.name}={param.default}" if default_exists else param.name
            comment: Optional[str] = None
            match status, default_exists:
                case ParamStatus.ALIVE, _:
                    pass
                case ParamStatus.KEPT, False:
                    snippet = yellow(snippet)
                    comment = f"will be ignored but {red('required')}"
                case (ParamStatus.PURGED, _) | (ParamStatus.KEPT, True):
                    snippet = red(strikethrough(snippet))
                    comment = "will be ignored and optional"
            snippet = f"{indent8}{snippet},"
            if comment is not None:
                snippet += f"  # <--- {comment}"
            return snippet

        def is_required(param: inspect.Parameter, status: ParamStatus) -> bool:
            return has_default(param) and status == ParamStatus.KEPT

        def has_default(param: inspect.Parameter) -> bool:
            return param.default is not inspect._empty

        params_snippet = "\n".join(as_snippet(param, param_status[name]) for name, param in original_params.items())
        code_snippet = f"{indent4}def forward(\n{indent8}self,\n{params_snippet}\n{indent4}):\n{indent8}..."
        extra_instruction = ""
        if any(is_required(param, param_status[name]) for name, param in original_params.items()):
            extra_instruction = (
                "However, you might still need to provide some of the dropped inputs to the converted model "
                "in order to synchronize the input signature with the original model. "
            )
        log.warning(
            "The converted model has dropped inputs (i.e. inputs that does not affect model's output(s).) "
            f"{extra_instruction}"
            "See the comments in the following code snippet for more details:\n"
            f"{code_snippet}"
        )

    if not all(status == ParamStatus.ALIVE for status in param_status.values()):
        warn_dropped_inputs()

    return graph_module

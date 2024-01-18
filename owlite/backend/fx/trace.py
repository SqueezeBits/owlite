# pylint: disable=protected-access
import inspect
import sys
import traceback
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Optional, Union

import torch
import torch._dynamo as torch_dynamo
from torch import Tensor
from torch.fx.graph_module import GraphModule, _WrappedCall
from torch.nn.parallel import DataParallel, DistributedDataParallel

from owlite_core.logger import log

from ...enums import OwLiteStatus
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


# pylint: disable-next=missing-function-docstring
def insert_output_adapter(graph_module: GraphModule, output: Any) -> GraphModule:
    graph = graph_module.graph
    output_node = find_the_output_node(graph)
    with graph.inserting_before(output_node):
        # create node target
        output_adapter_func = create_output_adapter(output)
        # create node
        adapter_node = graph.call_function(output_adapter_func, output_node.args, output_node.kwargs)
        # create new output node
        graph.output(adapter_node)

        # remove original output node
        graph.erase_node(output_node)
        # check if graph is valid, this is just for good measure
        graph.lint()
        # update graph module forward
        graph_module.recompile()
    return graph_module


# pylint: disable-next=missing-function-docstring
def create_output_adapter(original_output: Any) -> Callable[[Any], Any]:
    # pylint: disable-next=missing-class-docstring, too-few-public-methods
    class TensorPlaceholder:
        pass

    def extract_structure(original_output: Any) -> Any:
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

            return None

        return _extract_structure(original_output)

    extracted_output_structure = extract_structure(original_output)

    def output_adapter(graph_module_out: Any) -> Any:
        # output adapter node will be transparent to torch.onnx.export(jit.trace)
        if torch._C._get_tracing_state():  # pylint: disable=protected-access
            return graph_module_out

        # flatten graph module output
        flattened_outputs = []

        def _extract_output(output: Union[Iterable, torch.Tensor]) -> None:
            if isinstance(output, torch.Tensor):
                flattened_outputs.append(output)
            elif isinstance(output, Iterable):
                for out in output:
                    _extract_output(out)

        _extract_output(graph_module_out)

        # rearrange output to match with original structure
        def _create_return(obj: Any) -> Any:
            if isinstance(obj, TensorPlaceholder):
                return flattened_outputs.pop(0)
            if isinstance(obj, tuple):
                return tuple(_create_return(e) for e in obj)
            if isinstance(obj, list):
                return [_create_return(e) for e in obj]
            if isinstance(obj, OrderedDict):
                return OrderedDict({key: _create_return(val) for (key, val) in obj.items()})
            if isinstance(obj, dict):
                return {key: _create_return(val) for (key, val) in obj.items()}

            return tuple(flattened_outputs)

        return _create_return(extracted_output_structure)

    return output_adapter


# pylint: disable-next=missing-function-docstring
def patched_wrapped_call(self: Any, obj: GraphModule, *args: Any, **kwargs: Any) -> Any:
    params = OrderedDict(
        (k, v) for i, (k, v) in enumerate(inspect.signature(self.cls.forward).parameters.items()) if i > 0
    )

    if len(args) > len(params):
        if not obj.meta.get("has_warned_ignored_args", False):
            log.warning(
                f"The last {len(args) - len(params)} arguments given to "
                "the graph module's forward method will be ignored"
            )
            obj.meta["has_warned_ignored_args"] = True
        args = args[: len(params)]

    ignored_keys = [*filter(lambda key: key not in params, kwargs)]
    if ignored_keys:
        if not obj.meta.get("has_warned_ignored_kwargs", False):
            log.warning(
                "The following keyword arguments given to the graph module's forward method will be ignored: "
                f"{', '.join(ignored_keys)}"
            )
            obj.meta["has_warned_ignored_kwargs"] = True
        for key in ignored_keys:
            kwargs.pop(key)

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

    graph_module = apply_graph_module_transforms(graph_module)
    graph_module = insert_output_adapter(graph_module, output)

    original_params = inspect.signature(model.forward).parameters
    graph_module_params = inspect.signature(graph_module.forward).parameters

    ignored_params = OrderedDict(
        filter(
            lambda item: (
                item[0] not in graph_module_params
                and item[1].kind
                not in (
                    inspect._ParameterKind.VAR_POSITIONAL,
                    inspect._ParameterKind.VAR_KEYWORD,
                )
            ),
            original_params.items(),
        )
    )
    if ignored_params:
        log.warning(
            "The following parameters will be dropped from the graph module's forward method: "
            f"{', '.join(ignored_params)}"
        )
    graph_module.train(training_status)
    graph_module.meta["owlite_status"] = OwLiteStatus.NOT_COMPRESSED
    return graph_module

# pylint: disable=protected-access, too-many-statements
import inspect
from typing import Any

import torch
from torch.fx.graph_module import GraphModule
from torch.nn.parallel import DataParallel, DistributedDataParallel

from ...core.logger import log
from ...enums import ModelStatus
from ..signature import Signature
from .graph_checker import validate_procedure_calls
from .optimize import optimize


# pylint: disable-next=too-many-locals
def symbolic_trace(model: torch.nn.Module, *args: Any, **kwargs: Any) -> GraphModule:
    """Symbolically trace the input `model` to convert it into a GraphModule.

    In order for the tracing to be successful, the `model` must be able to pass `torch.compile(model, fullgraph=True)`.

    Args:
        model (torch.nn.Module): a torch.nn.Module instance.
        *args: the example input(s) that would be passed to the model's forward method.
        **kwargs: the example input(s) that would be passed to the model's forward method.

    Raises:
        TypeError: if the `model` is not an instance of `torch.nn.Module`
        RuntimeError: if the tracing fails.

    Returns:
        GraphModule: the converted GraphModule.
    """
    # If `owlite.fx.symbolic_trace` is called on more than models, the compilation caches created from one model
    # might cause an unexpected error for another. Furthermore, OwLite currently doesn't support graph breaks within
    # a model, there's no need to keep the compilation caches.
    # Hence we always clear all caches before running the compilation.
    torch._dynamo.reset()
    torch.compiler.reset()

    given_type = type(model)
    if isinstance(model, DataParallel | DistributedDataParallel):
        log.error(
            f"{given_type} is not supported by symbolic trace, please use 'attribute' module to unwrap model "
            f"from {given_type}. Try owlite.fx.symbolic_trace(model.module, ...)"
        )

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module instance but object of type {given_type} given: {model}")

    training_status = model.training

    original_signature = inspect.signature(model.forward)
    exporter = torch._dynamo.export(model, aten_graph=False, pre_dispatch=False, tracing_mode="real")
    graph_module = exporter(*args, **kwargs).graph_module

    graph_module.train(training_status)
    graph_module.meta["status"] = ModelStatus.TRACED
    graph_module_input_signature = Signature.from_module(graph_module, args, kwargs)
    graph_module_input_signature.warn_signature_change(dict(original_signature.parameters.items()))
    graph_module.meta["input_signature"] = graph_module_input_signature
    validate_procedure_calls(graph_module)

    _ = optimize(graph_module)

    return graph_module

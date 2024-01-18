import logging

from tabulate import tabulate
from torch.fx.graph_module import GraphModule

from owlite_core.logger import log

from ..utils import targetstr
from .node import get_target_module


def serialize(graph_module: GraphModule) -> str:
    r"""Exports a model into ONNX format.

    If ``model`` is not a :class:`torch.jit.ScriptModule` nor a
    :class:`torch.jit.ScriptFunction`, this runs
    ``model`` once in order to convert it to a TorchScript graph to be exported
    (the equivalent of :func:`torch.jit.trace`). Thus this has the same limited support
    for dynamic control flow as :func:`torch.jit.trace`.

    Args:
        model (:class:`torch.nn.Module`, :class:`torch.jit.ScriptModule` or :class:`torch.jit.ScriptFunction`):
            the model to be exported.

    Retruns:
        serialized (str): serialized fx graph
    """
    graph = graph_module.graph
    node_specs = [
        [
            n.op,
            n.name,
            targetstr(n.target)
            if n.op == "call_function"
            else get_target_module(n).__class__
            if n.op == "call_module"
            else n.target,
            n.args,
            n.kwargs,
        ]
        for n in graph.nodes
    ]

    serialized = tabulate(node_specs, headers=["opcode", "name", "target", "args", "kwargs"])

    if log.level <= logging.DEBUG:
        print(serialized)

    # add some encoding maybe?

    return serialized

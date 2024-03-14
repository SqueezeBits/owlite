import logging

from tabulate import tabulate
from torch.fx.graph_module import GraphModule

from ...owlite_core.logger import log
from ..utils import targetstr
from .node import get_target_module


def serialize(graph_module: GraphModule) -> str:
    """Serializes model into textual form.

    Args:
        model (GraphModule): the model to be serialized

    Returns:
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

    return serialized

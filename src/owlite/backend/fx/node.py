import inspect

import torch
from torch.fx.graph import Graph, Node

from ..utils import log
from .target import CONSTANT_TARGETS, NONDETERMINISTIC_TARGETS
from .types import TorchTarget


def find_placeholders(graph: Graph) -> list[Node]:
    """Find all placeholder nodes.

    Args:
        graph (Graph): the input graph

    Returns:
        list[Node]: the list of nodes whose op is "placeholder"
    """
    return [*filter(lambda n: n.op == "placeholder", graph.nodes)]


def find_the_output_node(graph: Graph) -> Node:
    """Find the unique output node in the graph.

    Args:
        graph (Graph): the input graph

    Raises:
        RuntimeError: if the graph has no output node or more than one output nodes.

    Returns:
        Node: the unique node whose op is "output"
    """
    outputs = [*filter(lambda n: n.op == "output", graph.nodes)]
    if len(outputs) == 0:
        raise RuntimeError('torch.fx.Graph has no node whose op == "output"')

    if len(outputs) > 1:
        raise RuntimeError('torch.fx.Graph has more than one node whose op == "output"')

    return outputs[0]


def find_constant_nodes(graph: Graph) -> list[Node]:
    """Find all constant-foldable nodes in the graph.

    Args:
        graph (Graph): the input graph

    Returns:
        list[Node]: the list containing all constant-foldable nodes in the graph.
    """
    constant_nodes: list[Node] = []
    non_constant_nodes: list[Node] = []

    def is_constant_getattr_node(node: Node) -> bool:
        if node.op == "get_attr":
            return True
        if not (node.op == "call_function" and node.target is getattr):
            return False
        name = node.args[1] if len(node.args) > 1 else node.kwargs.get("name", None)
        return name not in ("T",)

    def is_constant(node: Node) -> bool:
        if node in constant_nodes:
            return True
        if node in non_constant_nodes:
            return False

        if node.op == "placeholder":
            non_constant_nodes.append(node)
            return False

        if node.target in NONDETERMINISTIC_TARGETS:
            non_constant_nodes.append(node)
            return False

        is_getattr_node = is_constant_getattr_node(node)
        is_constant_generating_node = node.op in ("call_function", "call_method") and node.target in CONSTANT_TARGETS
        is_missing_input_node = len(node.all_input_nodes) == 0
        if any((is_getattr_node, is_missing_input_node, is_constant_generating_node)):
            constant_nodes.append(node)
            return True

        result: bool = all(is_constant(input_node) for input_node in node.all_input_nodes)
        if result:
            constant_nodes.append(node)
        else:
            non_constant_nodes.append(node)
        return result

    for node in graph.nodes:
        _ = is_constant(node)

    return constant_nodes


def is_output_adapter_node(node: Node) -> bool:
    """Check if the node is an output adapter node.

    Args:
        node: torch.fx.Node. Node to be checked

    Returns:
        bool: True, if node is output_adapter_node. Else, False.
    """
    return (
        inspect.isfunction(node.target)
        and getattr(node.target, "__name__", "") == "output_adapter"
        and getattr(inspect.getmodule(node.target), "__name__", "") == "owlite.backend.fx.trace"
    )


def get_target_module(node: Node) -> torch.nn.Module | None:
    """Find the module the node is targeting only when the node is a proper "call_module" node.

    Args:
        node (FXNode): a node.

    Returns:
        torch.nn.Module | None: the module that the node is pointing to if
            * its op is `"call_module"`; and
            * its target is a string; and
            * it belongs to a GraphModule instance
            Otherwise, `None` is returned
    """
    if node.op != "call_module" or not isinstance(node.target, str):
        return None
    graph_module = node.graph.owning_module
    if graph_module is None:
        return None
    module: torch.nn.Module | None = None
    try:
        module = graph_module.get_submodule(node.target)
    except AttributeError as e:
        log.warning(e)
        return None
    return module


def get_torch_target(node: Node) -> TorchTarget | None:
    """Get the PyTorch target module or function of the node.

    Args:
        node (Node): a node.

    Returns:
        TorchTarget | None:
            * If `node.op` is `"call_module"`, returns the class of module instance it is targeting.
            * If `node.op` is either `"call_function"` or `"call_method"` and its target is from torch,
            returns `node.target`
            * Otherwise, returns `None`
    """
    target_module = get_target_module(node)
    if target_module is not None:
        return type(target_module)
    if node.op == "call_method" and isinstance(node.target, str) and hasattr(torch.Tensor, node.target):
        return node.target
    if node.op == "call_function" and callable(node.target):
        return node.target
    return None

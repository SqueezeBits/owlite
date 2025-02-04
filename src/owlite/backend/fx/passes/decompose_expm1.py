import torch
from torch.fx.node import Node

from .rewrite_pass import RewritePass


class DecomposeExpm1(RewritePass):
    """Decompose all occurrences of `torch.expm1(x)` by `torch.exp(x) - 1`."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (node.op == "call_function" and node.target is torch.expm1):
            return {}

        graph = node.graph
        input_node = node.all_input_nodes[0]
        with graph.inserting_before(node):
            exp_node = graph.call_function(torch.exp, args=(input_node,))
            sub_node = graph.call_function(torch.sub, args=(exp_node, 1))
        return {node: sub_node}

import torch
import torch.nn.functional as F
from torch.fx.node import Node

from ..node import get_target_module
from .rewrite_pass import RewritePass


class DecomposeSiLU(RewritePass):
    """Decompose all occurrences of `torch.nn.SiLU` and `F.silu` by sigmoid and mul node pairs."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if not (
            isinstance(get_target_module(node), torch.nn.SiLU) or (node.op == "call_function" and node.target is F.silu)
        ):
            return {}

        graph = node.graph
        input_node = node.all_input_nodes[0]
        with graph.inserting_before(node):
            sigmoid_node = graph.call_function(F.sigmoid, args=(input_node,))
            mul_node = graph.call_function(torch.mul, args=(input_node, sigmoid_node))
        return {node: mul_node}

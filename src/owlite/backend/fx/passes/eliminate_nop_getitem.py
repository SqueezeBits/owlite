import operator

import torch
import torch.utils._pytree as pytree
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class EliminateNopGetitem(PassBase):
    """Eliminate function calls of operator.getitem with no effect."""

    def call(self, graph_module: GraphModule) -> PassResult:
        """Eliminate function calls of operator.getitem with no effect.

        Args:
            graph_module (GraphModule): the input graph module

        Returns:
            PassResult: the result of the pass
        """
        nodes: list[Node] = [*graph_module.graph.nodes]
        modified = False
        for node in nodes:
            if not (
                node.op == "call_function"
                and node.target is operator.getitem
                and len(node.args) == 2
                and isinstance(x := node.args[0] if node.args else node.kwargs.get("input"), Node)
                and isinstance(example_value := x.meta.get("example_value"), torch.Tensor)
                and not any(isinstance(leaf, Node) for leaf in pytree.tree_flatten(node.args[1])[0])
                and operator.getitem(  # type: ignore[misc]
                    torch.zeros(example_value.shape, dtype=example_value.dtype), node.args[1]
                ).shape
                == example_value.shape
            ):
                continue

            modified = modified or len(node.replace_all_uses_with(x)) > 0

        return PassResult(graph_module, modified)

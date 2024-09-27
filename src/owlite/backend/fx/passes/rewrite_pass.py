from abc import abstractmethod

from torch.fx import GraphModule
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class RewritePass(PassBase):
    """Abstract class for implementing node-wise rewriting pass."""

    def call(self, graph_module: GraphModule) -> PassResult:
        """Apply `cls.rewrite` method across all nodes in the graph.

        Args:
            graph_module (GraphModule): the input graph module

        Returns:
            PassResult: the result of the pass
        """
        modified = False
        nodes = list((graph := graph_module.graph).nodes)
        for node in nodes:
            if replacement_map := self.rewrite(node):
                is_replaced = [
                    len(existing_node.replace_all_uses_with(rewritten_node)) > 0
                    for existing_node, rewritten_node in replacement_map.items()
                ]
                modified = modified or any(is_replaced)
                graph.eliminate_dead_code()
                graph.lint()

        return PassResult(graph_module, modified)

    @classmethod
    @abstractmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        """Rewrite the given node.

        Args:
            node (Node): a node to rewrite

        Returns:
            dict[Node, Node]: a dictionary mapping an existing node to its replacement.
        """

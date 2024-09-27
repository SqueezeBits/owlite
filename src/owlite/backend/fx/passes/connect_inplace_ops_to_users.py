import operator

from torch.fx import GraphModule
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class ConnectInplaceOpsToUsers(PassBase):
    """Connect `call_function(operator.setitem)` nodes to their user nodes."""

    def call(self, graph_module: GraphModule) -> PassResult:
        """Connect `call_function(operator.setitem)` nodes to its user nodes.

        Args:
            graph_module (GraphModule): the input graph module

        Returns:
            PassResult: the result of the pass
        """
        nodes = list(graph_module.graph.nodes)

        def replace_users_with_index_larger_than(*, target: Node, replacement: Node, index_lower_bound: int) -> bool:
            def is_index_greater_than_lower_bound(user: Node) -> bool:
                try:
                    return nodes.index(user) > index_lower_bound
                except IndexError:
                    return False

            replaced_uses = target.replace_all_uses_with(replacement, delete_user_cb=is_index_greater_than_lower_bound)
            return len(replaced_uses) > 0

        modified = False
        for inplace_node_index, inplace_node in enumerate(nodes):
            if not (
                is_inplace(inplace_node)
                and isinstance(
                    (
                        parent_node := inplace_node.args[0]
                        if len(inplace_node.args) > 0
                        else inplace_node.kwargs.get("a")
                    ),
                    Node,
                )
            ):
                continue

            is_any_user_replaced = replace_users_with_index_larger_than(
                target=parent_node, replacement=inplace_node, index_lower_bound=inplace_node_index
            )
            modified = modified or is_any_user_replaced
        return PassResult(graph_module, modified)


def is_inplace(node: Node) -> bool:
    """Check if the given node is calling an inplace function or method.

    Args:
        node (Node): a node

    Returns:
        bool: `True` if it is calling an inplace function or method, `False` otherwise
    """
    if node.op == "call_function" and node.target in (operator.setitem,):
        return True
    if node.op == "call_method" and isinstance(node.target, str) and node.target.endswith("_"):
        return True
    return False

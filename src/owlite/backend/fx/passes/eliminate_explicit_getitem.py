import operator

from torch.fx import GraphModule
from torch.fx.node import Argument, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class EliminateExplicitGetitem(PassBase):
    """Eliminate function calls of operator.getitem on list / tuple / dict."""

    def call(self, graph_module: GraphModule) -> PassResult:
        """Eliminate function calls of operator.getitem on list / tuple / dict.

        Args:
            graph_module (GraphModule): the input graph module

        Returns:
            PassResult: the result of the pass
        """
        nodes: list[Node] = [*graph_module.graph.nodes]
        modified = False
        for node in nodes:
            if not (node.op == "call_function" and node.target is operator.getitem and len(node.args) == 2):
                continue

            container = node.args[0]
            position = node.args[1]
            value: Argument
            if isinstance(container, dict) and isinstance(position, str):
                value = container[position]
            elif isinstance(container, list | tuple) and isinstance(position, int | slice):
                value = container[position]
            else:
                continue

            usages: dict[Node, int | str] = {}
            for user in node.users:
                if node in user.args:
                    i = user.args.index(node)
                    usages[user] = i
                    continue
                for position, value in [*node.kwargs.items()]:
                    if value is node:
                        usages[user] = position
                        break

            modified = modified or len(usages) > 0
            for user, index_or_key in usages.items():
                if isinstance(index_or_key, int):
                    user.args = user.args[:index_or_key] + (value,) + user.args[index_or_key + 1 :]
                    continue
                user.kwargs[index_or_key] = value

        return PassResult(graph_module, modified)

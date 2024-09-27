import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from ..node import get_target_module


class EliminateIdentity(PassBase):
    """Eliminate module calls of torch.nn.Identity."""

    def call(self, graph_module: GraphModule) -> PassResult:
        """Eliminate module calls of torch.nn.Identity.

        Args:
            graph_module (GraphModule): the input graph module

        Returns:
            PassResult: the result of the pass
        """
        nodes: list[Node] = [*graph_module.graph.nodes]
        modified = False
        for node in nodes:
            if not (node.op == "call_module" and isinstance(get_target_module(node), torch.nn.Identity)):
                continue

            x = node.args[0] if node.args else node.kwargs.get("input")
            usages: dict[Node, int | str] = {}
            for user in node.users:
                if node in user.args:
                    i = user.args.index(node)
                    usages[user] = i
                    continue
                for key, value in [*node.kwargs.items()]:
                    if value is node:
                        usages[user] = key
                        break

            modified = modified or len(usages) > 0
            for user, index_or_key in usages.items():
                if isinstance(index_or_key, int):
                    user.args = user.args[:index_or_key] + (x,) + user.args[index_or_key + 1 :]
                    continue
                user.kwargs[index_or_key] = x

        return PassResult(graph_module, modified)

from types import EllipsisType

import torch
from torch.fx.node import Node

from .node_argument import NodeArgument
from .rewrite_pass import RewritePass


class ConcatNodeArgument(NodeArgument):
    """The arguments of a "call_function" node with target `torch.cat`, `torch.concat` or `torch.concatenate`."""

    tensors: tuple[Node, ...] | list[Node]
    dim: int | str | EllipsisType | None
    out: Node | None = None

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return node.op == "call_function" and node.target in (torch.cat, torch.concat, torch.concatenate)


class FuseConsecutiveConcats(RewritePass):
    """Fuse consecutive calls of torch.cat, torch.concat or torch.concatenate."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if (arguments := ConcatNodeArgument.extract_from(node)) is None:
            return {}

        graph = node.graph
        args: list[Node] = []
        has_parent_concat_with_same_dim = False
        for parent in arguments.tensors:
            if (
                parent_arguments := ConcatNodeArgument.extract_from(parent)
            ) is not None and parent_arguments.dim == arguments.dim:
                args.extend(parent_arguments.tensors)
                has_parent_concat_with_same_dim = True
            else:
                args.append(parent)
        if not has_parent_concat_with_same_dim:
            return {}

        with graph.inserting_before(node):
            fused_concat_node = graph.call_function(torch.cat, args=(tuple(args),), kwargs={"dim": arguments.dim})
        return {node: fused_concat_node}

import torch.nn.functional as F
from torch.fx.node import Node

from .node_argument import NodeArgument
from .rewrite_pass import RewritePass


class InProjectionNodeArgument(NodeArgument):
    """The arguments of a "call_function" node with target `F._in_projection`."""

    q: Node
    k: Node
    v: Node
    w_q: Node
    w_k: Node
    w_v: Node
    b_q: Node | None = None
    b_k: Node | None = None
    b_v: Node | None = None

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return (
            # pylint: disable-next=protected-access
            node.op == "call_function" and node.target is F._in_projection  # type: ignore
        )


class DecomposeInProjection(RewritePass):
    """Decompose all occurrences of `F._in_projection` by an equivalent subgraph.

    Note: this rewrite pass is implemented based on torch>=2.3.1,<=2.4.0
    """

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if (arguments := InProjectionNodeArgument.extract_from(node)) is None:
            return {}

        graph = node.graph
        with graph.inserting_before(node):
            q = graph.call_function(F.linear, (arguments.q, arguments.w_q, arguments.b_q))
            k = graph.call_function(F.linear, (arguments.k, arguments.w_k, arguments.b_k))
            v = graph.call_function(F.linear, (arguments.v, arguments.w_v, arguments.b_v))
            old_q, old_k, old_v = tuple(node.users)

        return {old_q: q, old_k: k, old_v: v}

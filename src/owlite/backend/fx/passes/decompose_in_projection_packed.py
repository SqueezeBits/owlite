import operator

import torch.nn.functional as F
from torch.fx.node import Node

from .node_argument import NodeArgument
from .rewrite_pass import RewritePass


class InProjectionPackedNodeArgument(NodeArgument):
    """The arguments of a "call_function" node with target `F._in_projection_packed`."""

    q: Node
    k: Node
    v: Node
    w: Node
    b: Node | None = None

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return (
            # pylint: disable-next=protected-access
            node.op == "call_function" and node.target is F._in_projection_packed  # type: ignore
        )


class DecomposeInProjectionPacked(RewritePass):
    """Decompose all occurrences of `F._in_projection_packed` by an equivalent subgraph.

    Note: this rewrite pass is implemented based on torch>=2.3.1,<=2.4.0
    """

    @classmethod  # pylint: disable-next=too-many-locals
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if (arguments := InProjectionPackedNodeArgument.extract_from(node)) is None:
            return {}

        graph = node.graph
        q, k, v, w, b = arguments.q, arguments.k, arguments.v, arguments.w, arguments.b
        with graph.inserting_before(node):
            embed_dim = graph.call_method("size", (q, -1))
            if k is v:
                if q is k:
                    # self-attention
                    proj = graph.call_function(F.linear, (q, w, b))
                    # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as
                    # chunk()
                    proj = graph.call_method("unflatten", (proj, -1, (3, embed_dim)))
                    proj = graph.call_method("unsqueeze", (proj, 0))
                    proj = graph.call_method("transpose", (proj, 0, -2))
                    proj = graph.call_method("squeeze", (proj, -2))
                    proj = graph.call_method("contiguous", (proj,))
                    new_q, new_k, new_v = (graph.call_function(operator.getitem, (proj, i)) for i in range(3))
                else:
                    # encoder-decoder attention
                    embed_dim_x_2 = graph.call_function(operator.mul, (embed_dim, 2))

                    def call_method_split(x: Node) -> tuple[Node, Node]:
                        x_splits = graph.call_method("split", (x, [embed_dim, embed_dim_x_2]))
                        x_q, x_kv = (graph.call_function(operator.getitem, (x_splits, i)) for i in range(2))
                        return x_q, x_kv

                    w_q, w_kv = call_method_split(w)
                    if b is None:
                        b_q = b_kv = None
                    else:
                        b_q, b_kv = call_method_split(b)
                    q_proj = graph.call_function(F.linear, (q, w_q, b_q))
                    kv_proj = graph.call_function(F.linear, (k, w_kv, b_kv))
                    # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as
                    # chunk()
                    kv_proj = graph.call_method("unflatten", (kv_proj, -1, (2, embed_dim)))
                    kv_proj = graph.call_method("unsqueeze", (kv_proj, 0))
                    kv_proj = graph.call_method("transpose", (kv_proj, 0, -2))
                    kv_proj = graph.call_method("squeeze", (kv_proj, -2))
                    kv_proj = graph.call_method("contiguous", (kv_proj,))
                    new_q = q_proj
                    new_k, new_v = (graph.call_function(operator.getitem, (kv_proj, i)) for i in range(2))
            else:

                def call_method_chunk(x: Node) -> tuple[Node, Node, Node]:
                    x_chunks = graph.call_method("chunk", (x, 3))
                    x_q, x_k, x_v = (graph.call_function(operator.getitem, (x_chunks, i)) for i in range(3))
                    return x_q, x_k, x_v

                w_q, w_k, w_v = call_method_chunk(w)
                if b is None:
                    b_q = b_k = b_v = None
                else:
                    b_q, b_k, b_v = call_method_chunk(b)
                new_q = graph.call_function(F.linear, (q, w_q, b_q))
                new_k = graph.call_function(F.linear, (k, w_k, b_k))
                new_v = graph.call_function(F.linear, (v, w_v, b_v))

        old_q, old_k, old_v = tuple(node.users)
        return {old_q: new_q, old_k: new_k, old_v: new_v}

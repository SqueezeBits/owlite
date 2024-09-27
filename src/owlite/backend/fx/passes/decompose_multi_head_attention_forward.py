import builtins
import math
import operator

import torch
import torch.nn.functional as F
from torch.fx import Graph
from torch.fx.node import Node

from .node_argument import NodeArgument
from .rewrite_pass import RewritePass
from .utils import call_canonical_mask


class MultiHeadAttentionForwardNodeArgument(NodeArgument):
    """The arguments of a "call_function" node with target `F.multi_head_attention_forward`."""

    query: Node
    key: Node
    value: Node
    embed_dim_to_check: int
    num_heads: int
    in_proj_weight: Node | None
    in_proj_bias: Node | None
    bias_k: Node | None
    bias_v: Node | None
    add_zero_attn: bool
    dropout_p: float
    out_proj_weight: Node
    out_proj_bias: Node | None
    training: bool = True
    key_padding_mask: Node | None = None
    need_weights: bool = True
    attn_mask: Node | None = None
    use_separate_proj_weight: bool = False
    q_proj_weight: Node | None = None
    k_proj_weight: Node | None = None
    v_proj_weight: Node | None = None
    static_k: Node | None = None
    static_v: Node | None = None
    average_attn_weights: bool = True
    is_causal: bool = False

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return node.op == "call_function" and node.target is F.multi_head_attention_forward


class DecomposeMultiHeadAttentionForward(RewritePass):
    """Decompose all occurrences of `F.multi_head_attention_forward` by an equivalent subgraph.

    Note: this rewrite pass is implemented based on torch>=2.3.1,<=2.4.0
    """

    @classmethod  # pylint: disable-next=too-many-locals, too-many-statements, too-many-branches
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if (arguments := MultiHeadAttentionForwardNodeArgument.extract_from(node)) is None:
            return {}

        graph = node.graph
        query, key, value = (arguments.query, arguments.key, arguments.value)
        with graph.inserting_before(node):
            # Note that there are control flows in `torch.nn.MultiheadAttention.forward``
            # depending on `is_batched := query.dim() == 3` which is a constant available only at runtime
            # (and hence not available via GraphModule.)
            # The idea is to eliminate the control flows depending on `is_batched`.
            # [Notation] N = batch, S = sequence length, E = embedding dimension
            # There are two possible cases for the shapes of the q, k, v:
            # (i) is_batched: (S, N, E)
            # (ii) not is_batched: (S, E)
            tgt_len = graph.call_method("size", (query, 0))
            embed_dim = graph.call_method("size", (query, -1))
            src_len = graph.call_method("size", (key, 0))

            # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
            # is batched, run the computation and before returning squeeze the
            # batch dimension so that the output doesn't carry this temporary batch dimension.
            # if is_batched: (S, N, E) -> (S, N, E)
            # else: (S, E) -> (S, 1, E)
            def call_method_reshape(x: Node, seq_len: Node, embed_size: Node | None = None) -> Node:
                if embed_size is None:
                    embed_size = graph.call_method("size", (x, -1))
                return graph.call_method("reshape", (x, seq_len, -1, embed_size))

            # The "is" properties between query, key and value must be preserved
            # in order to specialize into the correct graph construction control flows
            # in `DecomposeInProjection` and `DecomposeInProjectionPacked`
            query = call_method_reshape(query, tgt_len, embed_dim)
            if key is value:
                # Note: query is now a user node of arguments.query
                if arguments.query is key:
                    key = value = query
                else:
                    key = value = call_method_reshape(key, src_len)
            else:
                key, value = (call_method_reshape(x, src_len) for x in (key, value))

            bsz = graph.call_method("size", (query, 1))
            head_dim = graph.call_function(operator.floordiv, (embed_dim, arguments.num_heads))

            is_causal = arguments.is_causal
            dropout_p = arguments.dropout_p
            key_padding_mask = call_canonical_mask(
                mask=arguments.key_padding_mask,
                mask_name="key_padding_mask",
                other=arguments.attn_mask,
                other_name="attn_mask",
                target=arguments.query,
            )
            attn_mask: Node | None
            if is_causal and key_padding_mask is None and not arguments.need_weights:
                # when we have a kpm or need weights, we need attn_mask
                # Otherwise, we use the is_causal hint go as is_causal
                # indicator to SDPA.
                attn_mask = None
            else:
                attn_mask = call_canonical_mask(
                    mask=arguments.attn_mask,
                    mask_name="attn_mask",
                    other=None,
                    other_name="",
                    target=arguments.query,
                    check_other=False,
                )
                if key_padding_mask is not None:
                    is_causal = False

            # compute in-projection
            if not arguments.use_separate_proj_weight:
                assert (
                    arguments.in_proj_weight is not None
                ), "use_separate_proj_weight is False but in_proj_weight is None"
                in_projections = graph.call_function(
                    # pylint: disable-next=protected-access
                    F._in_projection_packed,  # type: ignore[attr-defined]
                    (query, key, value, arguments.in_proj_weight, arguments.in_proj_bias),
                )
            else:
                assert arguments.q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
                assert arguments.k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
                assert arguments.v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
                if arguments.in_proj_bias is None:
                    b_q = b_k = b_v = None
                else:
                    chunked_biases = graph.call_method("chunk", (arguments.in_proj_bias, 3))
                    b_q = graph.call_function(operator.getitem, (chunked_biases, 0))
                    b_k = graph.call_function(operator.getitem, (chunked_biases, 1))
                    b_v = graph.call_function(operator.getitem, (chunked_biases, 2))
                in_projections = graph.call_function(
                    # pylint: disable-next=protected-access
                    F._in_projection,  # type: ignore[attr-defined]
                    (
                        query,
                        key,
                        value,
                        arguments.q_proj_weight,
                        arguments.k_proj_weight,
                        arguments.v_proj_weight,
                        b_q,
                        b_k,
                        b_v,
                    ),
                )
            q = graph.call_function(operator.getitem, (in_projections, 0))
            k = graph.call_function(operator.getitem, (in_projections, 1))
            v = graph.call_function(operator.getitem, (in_projections, 2))

            # prep attention mask
            if attn_mask is not None:
                # if attn_mask.dim() == 2: (tgt_len, src_len) -> (1, tgt_len, src_len)
                # else: (bsz * num_heads, tgt_len, src_len) unchanged
                attn_mask = graph.call_method("reshape", (attn_mask, -1, tgt_len, src_len))

            # add bias along batch dimension (currently second)
            if (bias_k := arguments.bias_k) is not None and (bias_v := arguments.bias_v) is not None:
                assert arguments.static_k is None, "bias cannot be added to static key."
                assert arguments.static_v is None, "bias cannot be added to static value."
                k = graph.call_function(torch.cat, ([k, graph.call_method("repeat", (bias_k, 1, bsz, 1))],))
                v = graph.call_function(torch.cat, ([v, graph.call_method("repeat", (bias_v, 1, bsz, 1))],))
                if attn_mask is not None:
                    attn_mask = graph.call_function(F.pad, (attn_mask, (0, 1)))
                if key_padding_mask is not None:
                    key_padding_mask = graph.call_function(F.pad, (key_padding_mask, (0, 1)))

            # reshape q, k, v for multihead attention and make them batch first
            bsz_x_num_heads = graph.call_function(operator.mul, (bsz, arguments.num_heads))
            q = graph.call_method("view", (q, tgt_len, bsz_x_num_heads, head_dim))
            q = graph.call_method("transpose", (q, 0, 1))
            k = cls.kv_cache_handling(graph, k, arguments.static_k, bsz_x_num_heads, head_dim)
            v = cls.kv_cache_handling(graph, v, arguments.static_v, bsz_x_num_heads, head_dim)

            # add zero attention along batch dimension (now first)
            if arguments.add_zero_attn:
                k = graph.call_function(
                    torch.cat,
                    ([k, graph.call_method("new_zeros", (k, (bsz_x_num_heads, 1, head_dim)))],),
                    {"dim": 1},
                )
                v = graph.call_function(
                    torch.cat,
                    ([v, graph.call_method("new_zeros", (v, (bsz_x_num_heads, 1, head_dim)))],),
                    {"dim": 1},
                )
                if attn_mask is not None:
                    attn_mask = graph.call_function(F.pad, (attn_mask, (0, 1)))
                if key_padding_mask is not None:
                    key_padding_mask = graph.call_function(F.pad, (key_padding_mask, (0, 1)))

            # update source sequence length after adjustments
            src_len = graph.call_method("size", (k, 1))

            # merge key padding and attention masks
            if key_padding_mask is not None:
                key_padding_mask = graph.call_method("view", (key_padding_mask, bsz, 1, 1, src_len))
                key_padding_mask = graph.call_method("expand", (key_padding_mask, -1, arguments.num_heads, -1, -1))
                key_padding_mask = graph.call_method("reshape", (key_padding_mask, bsz_x_num_heads, 1, src_len))
                if attn_mask is None:
                    attn_mask = key_padding_mask
                else:
                    attn_mask = graph.call_function(operator.add, (attn_mask, key_padding_mask))

            # adjust dropout probability
            if not arguments.training:
                dropout_p = 0.0

            # (deep breath) calculate attention and out projection
            if arguments.need_weights:
                q_embed_dim = graph.call_method("size", (q, -1))
                scale = graph.call_function(math.sqrt, (graph.call_function(operator.truediv, (1.0, q_embed_dim)),))
                q_scaled = graph.call_function(operator.mul, (q, scale))
                k_transposed = graph.call_method("transpose", (k, -2, -1))

                assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

                if attn_mask is not None:
                    attn_output_weights = graph.call_function(torch.baddbmm, (attn_mask, q_scaled, k_transposed))
                else:
                    attn_output_weights = graph.call_function(torch.bmm, (q_scaled, k_transposed))
                attn_output_weights = graph.call_function(F.softmax, (attn_output_weights,), {"dim": -1})
                if dropout_p > 0.0:
                    attn_output_weights = graph.call_function(F.dropout, (attn_output_weights,), {"p": dropout_p})

                attn_output = graph.call_function(torch.bmm, (attn_output_weights, v))
                attn_output = graph.call_method("transpose", (attn_output, 0, 1))
                attn_output = graph.call_method("contiguous", (attn_output,))
                attn_output = graph.call_method("view", (attn_output, -1, embed_dim))
                attn_output = graph.call_function(
                    F.linear, (attn_output, arguments.out_proj_weight, arguments.out_proj_bias)
                )
                attn_output = graph.call_method("view_as", (attn_output, arguments.query))

                # optionally average attention weights over heads
                attn_output_weights = graph.call_method(
                    "view", (attn_output_weights, bsz, arguments.num_heads, tgt_len, src_len)
                )
                if arguments.average_attn_weights:
                    attn_output_weights = graph.call_method("mean", (attn_output_weights,), {"dim": 1})

                # See the docstring in `compute_attn_output_weights_shape`
                # for the explanation about why this part is different from the original implementation.
                input_query_shape = graph.call_function(builtins.getattr, (arguments.query, "shape"))
                w_shape = cls.compute_attn_output_weights_shape(
                    graph,
                    input_query_shape,
                    arguments.num_heads,
                    arguments.average_attn_weights,
                )
                attn_output_weights = graph.call_method("reshape", (attn_output_weights, w_shape))

                old_attn_output, old_attn_output_weights = tuple(node.users)
                return {
                    old_attn_output: attn_output,
                    old_attn_output_weights: attn_output_weights,
                }

            # attn_mask can be either (L,S) or (N*num_heads, L, S)
            # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
            # in order to match the input for SDPA of (N, num_heads, L, S)
            if attn_mask is not None:
                if key_padding_mask is attn_mask:
                    attn_mask = graph.call_method("view", (attn_mask, bsz, arguments.num_heads, 1, src_len))
                else:
                    attn_mask_numel = graph.call_method("numel", (attn_mask,))
                    tgt_len_x_src_len = graph.call_function(operator.mul, (tgt_len, src_len))
                    tgt_len_x_src_len_x_num_heads = graph.call_function(
                        operator.mul, (tgt_len_x_src_len, arguments.num_heads)
                    )
                    d0 = graph.call_function(operator.floordiv, (attn_mask_numel, tgt_len_x_src_len_x_num_heads))
                    d0 = graph.call_function(builtins.max, (d0, 1))
                    attn_mask = graph.call_method("view", (attn_mask, d0, -1, tgt_len, src_len))

            q = graph.call_method("view", (q, bsz, arguments.num_heads, tgt_len, head_dim))
            k, v = (graph.call_method("view", (x, bsz, arguments.num_heads, src_len, head_dim)) for x in (k, v))

            attn_output = graph.call_function(
                F.scaled_dot_product_attention, (q, k, v, attn_mask, dropout_p, is_causal)
            )
            attn_output = graph.call_method("permute", (attn_output, 2, 0, 1, 3))
            attn_output = graph.call_method("contiguous", (attn_output,))
            attn_output = graph.call_method("view", (attn_output, -1, embed_dim))

            attn_output = graph.call_function(
                F.linear, (attn_output, arguments.out_proj_weight, arguments.out_proj_bias)
            )
            attn_output = graph.call_method("view_as", (attn_output, arguments.query))
            old_attn_output = tuple(node.users)[0]
            return {old_attn_output: attn_output}

    @classmethod
    def kv_cache_handling(
        cls,
        graph: Graph,
        x: Node,
        static_x: Node | None,
        bsz_x_num_heads: Node,
        head_dim: Node,
    ) -> Node:
        """Resolve the KV cache and input KV.

        Args:
            graph (Graph): a graph
            x (Node): a node in the graph representing either input key (or value)
            static_x (Node | None): an optional node in the graph representing cached key (or value)
            bsz_x_num_heads (Node): the node in the graph generating the value `bsz * num_heads`
            head_dim (Node): the node in the graph generating the value `head_dim`

        Returns:
            Node: the resolved key (or value)
        """
        if static_x is None:
            x = graph.call_method("view", (x, -1, bsz_x_num_heads, head_dim))
            return graph.call_method("transpose", (x, 0, 1))
        # Skipping the assertions under the comment:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        # assuming the conditions are already met.
        return static_x

    @classmethod
    def compute_attn_output_weights_shape(
        cls,
        graph: Graph,
        input_query_shape: Node,
        num_heads: int,
        average_attn_weights: bool,
    ) -> Node:
        """Compute the shape of the output attention weights.

        This part does not exist in the original implementation of `F.multi_head_attention`, and handled
        with a control flow depending on the graph input's dimension.
        (See [torch/nn/functional.py at line 5494-5498 at tag v2.3.1](https://github.com/pytorch/pytorch/blob/63d5e9221bedd1546b7d364b5ce4171547db12a9/torch/nn/functional.py#L5494-L5498))
        However, FX graph cannot implement such dynamic control flow, requiring this workaround to compute the output
        attention mask shape, and then explicitly reshape the output attention mask by it.

        Args:
            graph (Graph): a graph
            input_query_shape (Node): the node encapsulating the `query.shape`
            num_heads (int): the argument `num_head` from `F.multi_head_attention_forward`
            average_attn_weights (bool): the argument `average_attn_weights` from `F.multi_head_attention_forward`

        Returns:
            Node: the computed shape node
        """
        # Note: `input_query_shape` is supposed to produce tuples of integers.
        # Namely, input_query_shape = (L, N, E) if is_batched else (L, E)
        # where:
        # - N := batch size
        # - L := query sequence length
        # - S := key/value sequence length (not explicitly present in the code, but represented by the -1)
        # The idea is to compute the output attention mask shape = (N, L, S) if is_batched else (L, S)
        # without the help of the dynamic control flow.

        # if is_batched: (L, N, E) -> (N, L)
        # else: (L, E) -> (L,)
        tgt_shape = graph.call_function(operator.getitem, (input_query_shape, slice(-2, -4, -1)))
        # if is_batched: (N, L) + (S,) -> (N, L, S)
        # else: (L,) + (S,) -> (L, S)
        w_shape = graph.call_function(operator.concat, (tgt_shape, (-1,)))
        if not average_attn_weights:
            # if is_batched: (N, L, S) -> (N, num_heads, L, S)
            # else: (L, S) -> (num_heads, L, S)
            w_batch = graph.call_function(operator.getitem, (w_shape, slice(None, -2)))
            w_mask_shape = graph.call_function(operator.getitem, (w_shape, slice(-2, None)))
            w_shape = graph.call_function(operator.concat, (w_batch, (num_heads,)))
            w_shape = graph.call_function(operator.concat, (w_shape, w_mask_shape))
        return w_shape

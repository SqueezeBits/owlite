# ruff: noqa: E501
import builtins
import math
import operator

import torch
import torch.nn.functional as F
from torch.fx.node import Node

from .node_argument import NodeArgument
from .rewrite_pass import RewritePass
from .utils import call_canonical_mask


class ScaledDotProductAttentionNodeArgument(NodeArgument):
    """The arguments of a "call_function" node with target `F.scaled_dot_product_attention`."""

    query: Node
    key: Node
    value: Node
    attn_mask: Node | None = None
    dropout_p: float = 0.0
    is_causal: bool = False
    scale: float | None = None

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return node.op == "call_function" and node.target is F.scaled_dot_product_attention


class DecomposeScaledDotProductAttention(RewritePass):
    """Decompose all occurrences of `F.scaled_dot_product_attention` by an equivalent subgraph.

    Note: this rewrite pass is implemented based on torch>=2.3.1,<=2.4.0
    """

    @classmethod  # pylint: disable-next=too-many-locals
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        """Rewrite node as an equivalent subgraph if it is a call_function(F.scaled_dot_product_attention).

        The reference code is from
        * [F.scaled_dot_product_attention in PyTorch 2.3](https://pytorch.org/docs/2.3/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention)
        * [F.scaled_dot_product_attention in PyTorch 2.4](https://pytorch.org/docs/2.4/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention) (Same as PyTorch 2.3)

        ```python
        # Efficient implementation equivalent to the following:
        def scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
        ) -> torch.Tensor:
            L, S = query.size(-2), key.size(-2)
            scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
            attn_bias = torch.zeros(L, S, dtype=query.dtype)
            if is_causal:
                assert attn_mask is None
                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                attn_bias.to(query.dtype)

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                else:
                    attn_bias += attn_mask
            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight += attn_bias
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            return attn_weight @ value
        ```

        Args:
            node (Node): a node to rewrite

        Returns:
            dict[Node, Node]: a dictionary mapping an existing node to its replacement.
        """
        if (arguments := ScaledDotProductAttentionNodeArgument.extract_from(node)) is None:
            return {}

        graph = node.graph
        query, key, value, attn_mask = arguments.query, arguments.key, arguments.value, arguments.attn_mask
        with graph.inserting_before(node):
            tgt_len, src_len = (graph.call_method("size", (x, -2)) for x in (query, key))
            device = graph.call_function(builtins.getattr, (query, "device"))
            dtype = graph.call_function(builtins.getattr, (query, "dtype"))
            scale_factor = arguments.scale
            if scale_factor is None:
                embed_dim = graph.call_method("size", (query, -1))
                sqrt_embed_dim = graph.call_function(math.sqrt, (embed_dim,))
                scale_factor = graph.call_function(operator.truediv, (1, sqrt_embed_dim))

            # Scale q, k before matmul for stability
            scale_factor = graph.call_function(math.sqrt, (scale_factor,))
            query = graph.call_function(operator.mul, (query, scale_factor))
            attn_bias: Node | None = None

            def initialize_attn_bias() -> Node:
                return graph.call_function(torch.zeros, (tgt_len, src_len), {"dtype": dtype, "device": device})

            if arguments.is_causal:
                assert attn_mask is None
                attn_bias = attn_bias or initialize_attn_bias()
                temp_mask = graph.call_function(torch.ones, (tgt_len, src_len), {"dtype": torch.bool, "device": device})
                temp_mask = graph.call_method("tril", (temp_mask,), {"diagonal": 0})
                temp_mask = graph.call_method("logical_not", (temp_mask,))
                attn_bias = graph.call_method("masked_fill_", (attn_bias, temp_mask, float("-inf")))
                attn_bias = graph.call_method("to", (attn_bias, dtype))

            if attn_mask is not None:
                attn_bias = attn_bias or initialize_attn_bias()
                # Workaround for the control flow depending on `attn_mask.dtype`:
                # Just make the attn_mask's dtype to `query.dtype` using `F._canonical_mask`
                # and then stick to the else branch in the reference code
                attn_mask = call_canonical_mask(
                    mask=attn_mask,
                    mask_name="attn_mask",
                    other=attn_bias,
                    other_name="attn_bias",
                    target=query,
                    check_other=True,
                )
                attn_bias = graph.call_function(operator.add, (attn_bias, attn_mask))

            key_transpose = graph.call_method("transpose", (key, -2, -1))
            key_transpose = graph.call_function(operator.mul, (key_transpose, scale_factor))
            attn_weight = graph.call_function(operator.matmul, (query, key_transpose))
            # Do not add `attn_bias` when it is zero
            if attn_bias is not None:
                attn_weight = graph.call_function(operator.add, (attn_weight, attn_bias))
            attn_weight = graph.call_function(torch.softmax, (attn_weight,), {"dim": -1})
            # When `F.scaled_dot_product_attention` is called from `F.multi_head_attention_forward`,
            # this guess will do the job. However, there's no way to figure out the exact value for
            # `train` parameter of `torch.dropout` in more general context.
            maybe_training = arguments.dropout_p > 0
            attn_weight = graph.call_function(
                torch.dropout, (attn_weight, arguments.dropout_p), {"train": maybe_training}
            )
            output = graph.call_function(operator.matmul, (attn_weight, value))

        return {node: output}

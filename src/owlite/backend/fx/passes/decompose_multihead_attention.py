import operator
from functools import cached_property

import torch
import torch.nn.functional as F
from torch.fx import Graph
from torch.fx.node import Argument, Node

from ..node import get_target_module
from .node_argument import NodeArgument
from .rewrite_pass import RewritePass


class MultiheadAttentionNodeArgument(NodeArgument):
    """The arguments of a "call_module" node with target module of type `torch.nn.MultiheadAttention`."""

    query: Node
    key: Node
    value: Node
    key_padding_mask: Node | None = None
    need_weights: bool = True
    attn_mask: Node | None = None
    average_attn_weights: bool = True
    is_causal: bool = False

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return node.op == "call_module" and isinstance(get_target_module(node), torch.nn.MultiheadAttention)

    @cached_property
    def module(self) -> torch.nn.MultiheadAttention:
        """The `torch.nn.MultiheadAttention` layer called by the node."""
        assert isinstance((m := get_target_module(self.node)), torch.nn.MultiheadAttention)
        return m


class DecomposeMultiheadAttention(RewritePass):
    """Decompose all occurrences of `torch.nn.MultiheadAttention` by an equivalent subgraph.

    Note: this rewrite pass is implemented based on torch>=2.3.1,<=2.4.0
    """

    @classmethod  # pylint: disable-next=too-many-locals
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if (arguments := MultiheadAttentionNodeArgument.extract_from(node)) is None:
            return {}

        graph = node.graph
        query, key, value = arguments.query, arguments.key, arguments.value
        with graph.inserting_before(node):
            # Note that there are control flows in `torch.nn.MultiheadAttention.forward``
            # depending on `is_batched := query.dim() == 3` which is a constant available only at runtime
            # (and hence not available via GraphModule.)
            # The idea is to eliminate the control flows depending on `is_batched` and only use control flows
            # depending on the static constant `self.batch_first`
            # [Notation] N = batch, S = sequence length, E = embedding dimension
            # There are three possible cases for the shapes of the q, k, v:
            # (i) is_batched and batch_first: (N, S, E)
            # (ii) is_batched and not batch_first: (S, N, E)
            # (iii) not is_batched: (S, E)
            if arguments.module.batch_first:
                # if is_batched: (N, S, E) -> (S, N, E)
                # else: (S, E) -> (S, E)
                # make sure that the transpose op does not affect the "is" property
                if key is value:
                    if query is key:
                        query = key = value = graph.call_method("transpose", (query, 0, -2))
                    else:
                        query, key = (graph.call_method("transpose", (x, 0, -2)) for x in (query, key))
                        value = key
                else:
                    query, key, value = (
                        graph.call_method("transpose", (x, 0, -2))
                        for x in (arguments.query, arguments.key, arguments.value)
                    )

            def may_get_attr(
                graph: Graph,
                relevant_module_attr: torch.Tensor | torch.nn.Module | None,
                name: str,
            ) -> Node | None:
                if relevant_module_attr is None:
                    return None
                return graph.get_attr(name)

            in_proj_weight = may_get_attr(graph, arguments.module.in_proj_weight, f"{node.target}.in_proj_weight")
            in_proj_bias = may_get_attr(graph, arguments.module.in_proj_bias, f"{node.target}.in_proj_bias")
            bias_k = may_get_attr(graph, arguments.module.bias_k, f"{node.target}.bias_k")
            bias_v = may_get_attr(graph, arguments.module.bias_v, f"{node.target}.bias_v")
            out_proj = graph.get_attr(f"{node.target}.out_proj")
            out_proj_weight = graph.get_attr(f"{out_proj.target}.weight")
            out_proj_bias = may_get_attr(graph, arguments.module.out_proj.bias, f"{out_proj.target}.bias")

            mha_forward_args: tuple[Argument, ...] = (
                query,
                key,
                value,
                arguments.module.embed_dim,
                arguments.module.num_heads,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                arguments.module.add_zero_attn,
                arguments.module.dropout,
                out_proj_weight,
                out_proj_bias,
            )
            mha_forward_common_kwargs: dict[str, Argument] = {
                "training": arguments.module.training,
                "key_padding_mask": arguments.key_padding_mask,
                "need_weights": arguments.need_weights,
                "attn_mask": arguments.attn_mask,
                "average_attn_weights": arguments.average_attn_weights,
                "is_causal": arguments.is_causal,
            }

            # pylint: disable-next=protected-access
            if not arguments.module._qkv_same_embed_dim:
                q_proj_weight = graph.get_attr(f"{node.target}.q_proj_weight")
                k_proj_weight = graph.get_attr(f"{node.target}.k_proj_weight")
                v_proj_weight = graph.get_attr(f"{node.target}.v_proj_weight")
                mha_forward_outputs = graph.call_function(
                    F.multi_head_attention_forward,
                    mha_forward_args,
                    {
                        **mha_forward_common_kwargs,
                        "use_separate_proj_weight": True,
                        "q_proj_weight": q_proj_weight,
                        "k_proj_weight": k_proj_weight,
                        "v_proj_weight": v_proj_weight,
                    },
                )
            else:
                mha_forward_outputs = graph.call_function(
                    F.multi_head_attention_forward,
                    mha_forward_args,
                    mha_forward_common_kwargs,
                )
            # attn_output: (S, E) if unbatched, otherwise (S, N, E)
            attn_output = graph.call_function(operator.getitem, (mha_forward_outputs, 0))
            if arguments.need_weights:
                attn_output_weights = graph.call_function(operator.getitem, (mha_forward_outputs, 1))
            else:
                attn_output_weights = None

            if arguments.module.batch_first:
                # if is_batched: (S, N, E) -> (N, S, E)
                # else: (S, E) -> (S, E)
                attn_output = graph.call_method("transpose", (attn_output, 0, -2))

        if arguments.need_weights and attn_output_weights is not None:
            old_attn_output, old_attn_output_weights = tuple(node.users)
            return {
                old_attn_output: attn_output,
                old_attn_output_weights: attn_output_weights,
            }

        old_attn_output = tuple(node.users)[0]
        return {old_attn_output: attn_output}

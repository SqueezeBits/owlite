# pylint: disable=duplicate-code
import torch
from torch.fx.node import Node

from ..node import get_target_module
from .node_argument import NodeArgument
from .rewrite_pass import RewritePass


class TransformerNodeArgument(NodeArgument):
    """The arguments of a "call_module" node with target module of type `torch.nn.Transformer`."""

    src: Node
    tgt: Node
    src_mask: Node | None = None
    tgt_mask: Node | None = None
    memory_mask: Node | None = None
    src_key_padding_mask: Node | None = None
    tgt_key_padding_mask: Node | None = None
    memory_key_padding_mask: Node | None = None
    src_is_causal: bool | None = None
    tgt_is_causal: bool | None = None
    memory_is_causal: bool = False

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return node.op == "call_module" and isinstance(get_target_module(node), torch.nn.Transformer)


class DecomposeTransformer(RewritePass):
    """Decompose all occurrences of `torch.nn.Transformer` by an equivalent subgraph.

    Note: this rewrite pass is implemented based on torch>=2.3.1,<=2.4.0
    """

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if (arguments := TransformerNodeArgument.extract_from(node)) is None:
            return {}

        if arguments.src_is_causal is None:
            raise NotImplementedError(
                "Found a `torch.nn.Transformer` layer forwarded with `src_is_causal=None`. "
                "OwLite cannot handle dynamic control flow triggered by `src_is_causal` detection. "
                "Please set its value to either `True` or `False`."
            )  # UX

        if arguments.tgt_is_causal is None:
            raise NotImplementedError(
                "Found a `torch.nn.Transformer` layer forwarded with `tgt_is_causal=None`. "
                "OwLite cannot handle dynamic control flow triggered by `tgt_is_causal` detection. "
                "Please set its value to either `True` or `False`."
            )  # UX

        graph = node.graph
        with graph.inserting_before(node):
            memory = graph.call_module(
                f"{node.target}.encoder",
                args=(arguments.src,),
                kwargs={
                    "mask": arguments.src_mask,
                    "src_key_padding_mask": arguments.src_key_padding_mask,
                    "is_causal": arguments.src_is_causal,
                },
            )
            output = graph.call_module(
                f"{node.target}.decoder",
                args=(arguments.tgt, memory),
                kwargs={
                    "tgt_mask": arguments.tgt_mask,
                    "memory_mask": arguments.memory_mask,
                    "tgt_key_padding_mask": arguments.tgt_key_padding_mask,
                    "memory_key_padding_mask": arguments.memory_key_padding_mask,
                    "tgt_is_causal": arguments.tgt_is_causal,
                    "memory_is_causal": arguments.memory_is_causal,
                },
            )
        return {node: output}

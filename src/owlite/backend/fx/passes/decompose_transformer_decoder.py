from functools import cached_property

import torch
from torch.fx.node import Node

from ..node import get_target_module
from .node_argument import NodeArgument
from .rewrite_pass import RewritePass


class TransformerDecoderNodeArgument(NodeArgument):
    """The arguments of a "call_module" node with target module of type `torch.nn.Transformer`."""

    tgt: Node
    memory: Node
    tgt_mask: Node | None = None
    memory_mask: Node | None = None
    tgt_key_padding_mask: Node | None = None
    memory_key_padding_mask: Node | None = None
    tgt_is_causal: bool | None = None
    memory_is_causal: bool = False

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return node.op == "call_module" and isinstance(get_target_module(node), torch.nn.TransformerDecoder)

    @cached_property
    def module(self) -> torch.nn.TransformerDecoder:
        """The `torch.nn.TransformerDecoder` layer called by the node."""
        assert isinstance((m := get_target_module(self.node)), torch.nn.TransformerDecoder)
        return m


class DecomposeTransformerDecoder(RewritePass):
    """Decompose all occurrences of `torch.nn.TransformerEncoder` by an equivalent subgraph.

    Note: this rewrite pass is implemented based on torch>=2.3.1,<=2.4.0
    """

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if (arguments := TransformerDecoderNodeArgument.extract_from(node)) is None:
            return {}

        if arguments.tgt_is_causal is None:
            raise NotImplementedError(
                "Found a `torch.nn.TransformerDecoder` layer forwarded with `tgt_is_causal=None`. "
                "OwLite cannot handle dynamic control flow triggered by `tgt_is_causal` detection. "
                "Please set its value to either `True` or `False`."
            )  # UX

        self_layers = arguments.module.layers
        # if not isinstance((first_layer := self_layers[0]), torch.nn.TransformerDecoderLayer):
        #     return {}

        graph = node.graph
        with graph.inserting_before(node):
            # Note: `tgt_is_causal` must not be a Node ...
            # seq_len = inline_get_seq_len(arguments.tgt, first_layer.self_attn.batch_first)
            # tgt_is_causal = graph.call_function(
            #     torch.nn.modules.transformer._detect_is_causal_mask,
            #     (arguments.tgt_mask, arguments.tgt_is_causal, seq_len)
            # )

            output = arguments.tgt
            for i in range(len(self_layers)):
                output = graph.call_module(
                    f"{node.target}.layers.{i}",
                    args=(output, arguments.memory),
                    kwargs={
                        "tgt_mask": arguments.tgt_mask,
                        "memory_mask": arguments.memory_mask,
                        "tgt_key_padding_mask": arguments.tgt_key_padding_mask,
                        "memory_key_padding_mask": arguments.memory_key_padding_mask,
                        "tgt_is_causal": arguments.tgt_is_causal,
                        "memory_is_causal": arguments.memory_is_causal,
                    },
                )

            if arguments.module.norm is not None:
                output = graph.call_module(f"{node.target}.norm", (output,))

        return {node: output}

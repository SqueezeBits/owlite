from functools import cached_property

import torch
from torch.fx.node import Node

from ..node import get_target_module
from .node_argument import NodeArgument
from .rewrite_pass import RewritePass
from .utils import call_canonical_mask


class TransformerEncoderNodeArgument(NodeArgument):
    """The arguments of a "call_module" node with target module of type `torch.nn.Transformer`."""

    src: Node
    mask: Node | None = None
    src_key_padding_mask: Node | None = None
    is_causal: bool | None = None

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return node.op == "call_module" and isinstance(get_target_module(node), torch.nn.TransformerEncoder)

    @cached_property
    def module(self) -> torch.nn.TransformerEncoder:
        """The `torch.nn.TransformerEncoder` layer called by the node."""
        assert isinstance((m := get_target_module(self.node)), torch.nn.TransformerEncoder)
        return m


class DecomposeTransformerEncoder(RewritePass):
    """Decompose all occurrences of `torch.nn.TransformerEncoder` by an equivalent subgraph.

    Note: this rewrite pass is implemented based on torch>=2.3.1,<=2.4.0
    """

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if (arguments := TransformerEncoderNodeArgument.extract_from(node)) is None:
            return {}

        if arguments.is_causal is None:
            raise NotImplementedError(
                "Found a `torch.nn.TransformerEncoder` layer forwarded with `is_causal=None`. "
                "OwLite cannot handle dynamic control flow triggered by `is_causal` detection. "
                "Please set its value to either `True` or `False`."
            )  # UX

        graph = node.graph
        with graph.inserting_before(node):
            src_key_padding_mask = call_canonical_mask(
                mask=arguments.src_key_padding_mask,
                mask_name="src_key_padding_mask",
                other=arguments.mask,
                other_name="mask",
                target=arguments.src,
            )

            mask = call_canonical_mask(
                mask=arguments.mask,
                mask_name="mask",
                other=None,
                other_name="",
                target=arguments.src,
                check_other=False,
            )

            self_layers = arguments.module.layers
            # first_layer = self_layers[0]
            # batch_first = first_layer.self_attn.batch_first

            # Note: `is_causal` must not be a node ...
            # seq_len = inline_get_seq_len(arguments.src, batch_first)
            # is_causal = graph.call_function(
            #     torch.nn.modules.transformer._detect_is_causal_mask,
            #     (mask, arguments.is_causal, seq_len),
            # )

            output = arguments.src
            for i in range(len(self_layers)):
                output = graph.call_module(
                    f"{node.target}.layers.{i}",
                    args=(output,),
                    kwargs={
                        "src_mask": mask,
                        "is_causal": arguments.is_causal,
                        "src_key_padding_mask": src_key_padding_mask,
                    },
                )

            if arguments.module.norm is not None:
                output = graph.call_module(f"{node.target}.norm", (output,))

        return {node: output}

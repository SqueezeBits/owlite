# pylint: disable=duplicate-code
import operator
from collections.abc import Callable
from functools import cached_property

import torch
from torch.fx.node import Node

from ..node import get_target_module
from .node_argument import NodeArgument
from .rewrite_pass import RewritePass
from .utils import call_canonical_mask


class TransformerEncoderLayerNodeArgument(NodeArgument):
    """The arguments of a "call_module" node with target module of type `torch.nn.Transformer`."""

    src: Node
    src_mask: Node | None = None
    src_key_padding_mask: Node | None = None
    is_causal: Node | bool

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return node.op == "call_module" and isinstance(get_target_module(node), torch.nn.TransformerEncoderLayer)

    @cached_property
    def module(self) -> torch.nn.TransformerEncoderLayer:
        """The `torch.nn.TransformerEncoderLayer` layer called by the node."""
        assert isinstance((m := get_target_module(self.node)), torch.nn.TransformerEncoderLayer)
        return m


class DecomposeTransformerEncoderLayer(RewritePass):
    """Decompose all occurrences of `torch.nn.TransformerEncoderLayer` by an equivalent subgraph.

    Note: this rewrite pass is implemented based on torch>=2.3.1,<=2.4.0
    """

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if (arguments := TransformerEncoderLayerNodeArgument.extract_from(node)) is None:
            return {}

        graph = node.graph
        with graph.inserting_before(node):
            src_key_padding_mask = call_canonical_mask(
                mask=arguments.src_key_padding_mask,
                mask_name="src_key_padding_mask",
                other=arguments.src_mask,
                other_name="src_mask",
                target=arguments.src,
            )

            src_mask = call_canonical_mask(
                mask=arguments.src_mask,
                mask_name="src_mask",
                other=None,
                other_name="",
                target=arguments.src,
                check_other=False,
            )

            x = arguments.src
            if arguments.module.norm_first:
                x = graph.call_function(
                    operator.add,
                    (
                        x,
                        cls.inline_sa_block(
                            node,
                            graph.call_module(f"{node.target}.norm1", (x,)),
                            src_mask,
                            src_key_padding_mask,
                            is_causal=arguments.is_causal,
                        ),
                    ),
                )
                x = graph.call_function(
                    operator.add,
                    (
                        x,
                        cls.call_ff_block(
                            node,
                            graph.call_module(f"{node.target}.norm2", (x,)),
                            arguments.module.activation,
                        ),
                    ),
                )
            else:
                x = graph.call_module(
                    f"{node.target}.norm1",
                    (
                        graph.call_function(
                            operator.add,
                            (
                                x,
                                cls.inline_sa_block(
                                    node,
                                    x,
                                    src_mask,
                                    src_key_padding_mask,
                                    is_causal=arguments.is_causal,
                                ),
                            ),
                        ),
                    ),
                )
                x = graph.call_module(
                    f"{node.target}.norm2",
                    (
                        graph.call_function(
                            operator.add,
                            (
                                x,
                                cls.call_ff_block(
                                    node,
                                    x,
                                    arguments.module.activation,
                                ),
                            ),
                        ),
                    ),
                )

        return {node: x}

    @classmethod
    def inline_sa_block(
        cls,
        node: Node,
        x: Node,
        attn_mask: Node | None,
        key_padding_mask: Node | None,
        is_causal: Node | bool,
    ) -> Node:
        """Inline the method `_sa_block` of `torch.nn.TransformerEncoderLayer`.

        Args:
            node (Node): a node corresponding to the argument `self` of the method `_sa_block`.
            x (Node): a node corresponding to the argument `x: Tensor` of the method `_sa_block`.
            attn_mask (Node | None): a node corresponding to the argument `attn_mask: Tensor | None`
                of the method `_sa_block`.
            key_padding_mask (Node | None): a node corresponding to the argument `key_padding_mask: Tensor | None`
                of the method `_sa_block`.
            is_causal (Node | bool): a node corresponding to the argument `is_causal: bool` of the method `_sa_block`.

        Returns:
            Node: the output node produced by the inlined call of the method `_sa_block`.
        """
        graph = node.graph
        x = graph.call_module(
            f"{node.target}.self_attn",
            args=(x, x, x),
            kwargs={
                "attn_mask": attn_mask,
                "key_padding_mask": key_padding_mask,
                "need_weights": False,
                "is_causal": is_causal,
            },
        )
        x = graph.call_function(operator.getitem, (x, 0))
        return graph.call_module(f"{node.target}.dropout1", (x,))

    @classmethod
    def call_ff_block(cls, node: Node, x: Node, activation: Callable[[torch.Tensor], torch.Tensor]) -> Node:
        """Inline the method `_ff_block` of `torch.nn.TransformerEncoderLayer`.

        Args:
            node (Node): a node corresponding to the argument `self` of the method `_ff_block`.
            x (Node): a node corresponding to the argument `x: Tensor` of the method `_ff_block`.
            activation (Callable[[torch.Tensor], torch.Tensor]): the activation function corresponding to
                `self.activation` used in the method `_ff_block`.

        Returns:
            Node: the output node produced by the inlined call of the method `_ff_block`.
        """
        graph = node.graph
        x = graph.call_module(f"{node.target}.linear1", (x,))
        x = graph.call_function(activation, (x,))
        x = graph.call_module(f"{node.target}.dropout", (x,))
        x = graph.call_module(f"{node.target}.linear2", (x,))
        return graph.call_module(f"{node.target}.dropout2", (x,))

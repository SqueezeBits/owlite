# pylint: disable=duplicate-code
import operator
from collections.abc import Callable
from functools import cached_property

import torch
from torch.fx.node import Node

from ..node import get_target_module
from .node_argument import NodeArgument
from .rewrite_pass import RewritePass


class TransformerDecoderLayerNodeArgument(NodeArgument):
    """The arguments of a "call_module" node with target module of type `torch.nn.Transformer`."""

    tgt: Node
    memory: Node
    tgt_mask: Node | None = None
    memory_mask: Node | None = None
    tgt_key_padding_mask: Node | None = None
    memory_key_padding_mask: Node | None = None
    tgt_is_causal: bool
    memory_is_causal: bool

    @classmethod
    def validate_node(cls, node: Node) -> bool:
        return node.op == "call_module" and isinstance(get_target_module(node), torch.nn.TransformerDecoderLayer)

    @cached_property
    def module(self) -> torch.nn.TransformerDecoderLayer:
        """The `torch.nn.TransformerDecoderLayer` layer called by the node."""
        assert isinstance((m := get_target_module(self.node)), torch.nn.TransformerDecoderLayer)
        return m


class DecomposeTransformerDecoderLayer(RewritePass):
    """Decompose all occurrences of `torch.nn.TransformerDecoderLayer` by an equivalent subgraph.

    Note: this rewrite pass is implemented based on torch>=2.3.1,<=2.4.0
    """

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        if (arguments := TransformerDecoderLayerNodeArgument.extract_from(node)) is None:
            return {}

        graph = node.graph
        with graph.inserting_before(node):
            x = arguments.tgt
            if arguments.module.norm_first:
                x = graph.call_function(
                    operator.add,
                    (
                        x,
                        cls.inline_sa_block(
                            node,
                            graph.call_module(f"{node.target}.norm1", (x,)),
                            arguments.tgt_mask,
                            arguments.tgt_key_padding_mask,
                            arguments.tgt_is_causal,
                        ),
                    ),
                )
                x = graph.call_function(
                    operator.add,
                    (
                        x,
                        cls.inline_mha_block(
                            node,
                            graph.call_module(f"{node.target}.norm2", (x,)),
                            arguments.memory,
                            arguments.memory_mask,
                            arguments.memory_key_padding_mask,
                            arguments.memory_is_causal,
                        ),
                    ),
                )
                x = graph.call_function(
                    operator.add,
                    (
                        x,
                        cls.inline_ff_block(
                            node,
                            graph.call_module(f"{node.target}.norm3", (x,)),
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
                                    arguments.tgt_mask,
                                    arguments.tgt_key_padding_mask,
                                    arguments.tgt_is_causal,
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
                                cls.inline_mha_block(
                                    node,
                                    x,
                                    arguments.memory,
                                    arguments.memory_mask,
                                    arguments.memory_key_padding_mask,
                                    arguments.memory_is_causal,
                                ),
                            ),
                        ),
                    ),
                )
                x = graph.call_module(
                    f"{node.target}.norm3",
                    (
                        graph.call_function(
                            operator.add,
                            (
                                x,
                                cls.inline_ff_block(
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
        is_causal: bool,
    ) -> Node:
        """Inline the method `_sa_block` of `torch.nn.TransformerDecoderLayer`.

        Args:
            node (Node): a node corresponding to the argument `self` of the method `_sa_block`.
            x (Node): a node corresponding to the argument `x: Tensor` of the method `_sa_block`.
            attn_mask (Node | None): a node corresponding to the argument `attn_mask: Tensor | None`
                of the method `_sa_block`.
            key_padding_mask (Node | None): a node corresponding to the argument `key_padding_mask: Tensor | None`
                of the method `_sa_block`.
            is_causal (bool): a node corresponding to the argument `is_causal: bool` of the method `_sa_block`.

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
                "is_causal": is_causal,
                "need_weights": False,
            },
        )
        x = graph.call_function(operator.getitem, (x, 0))
        return graph.call_module(f"{node.target}.dropout1", (x,))

    @classmethod
    def inline_mha_block(
        cls,
        node: Node,
        x: Node,
        mem: Node,
        attn_mask: Node | None,
        key_padding_mask: Node | None,
        is_causal: bool,
    ) -> Node:
        """Inline the method `_mha_block` of `torch.nn.TransformerDecoderLayer`.

        Args:
            node (Node): a node corresponding to the argument `self` of the method `_mha_block`.
            x (Node): a node corresponding to the argument `x: Tensor` of the method `_mha_block`.
            mem (Node): a node corresponding to the argument `mem: Tensor` of the method `_mha_block`.
            attn_mask (Node | None): a node corresponding to the argument `attn_mask: Tensor | None`
                of the method `_mha_block`.
            key_padding_mask (Node | None): a node corresponding to the argument `key_padding_mask: Tensor | None`
                of the method `_mha_block`.
            is_causal (bool): a node corresponding to the argument `is_causal: bool` of the method `_mha_block`.

        Returns:
            Node: the output node produced by the inlined call of the method `_mha_block`.
        """
        graph = node.graph
        x = graph.call_module(
            f"{node.target}.multihead_attn",
            args=(x, mem, mem),
            kwargs={
                "attn_mask": attn_mask,
                "key_padding_mask": key_padding_mask,
                "is_causal": is_causal,
                "need_weights": False,
            },
        )
        x = graph.call_function(operator.getitem, (x, 0))
        return graph.call_module(f"{node.target}.dropout2", (x,))

    @classmethod
    def inline_ff_block(
        cls,
        node: Node,
        x: Node,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> Node:
        """Inline the method `_ff_block` of `torch.nn.TransformerDecoderLayer`.

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
        return graph.call_module(f"{node.target}.dropout3", (x,))

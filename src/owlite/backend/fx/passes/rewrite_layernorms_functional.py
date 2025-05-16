import torch
import torch.nn.functional as F
from torch.fx.node import Node

from ....core.logger import log
from ..node import get_target_module
from .rewrite_pass import RewritePass


class RewriteLayerNormsFunctional(RewritePass):
    """Rewrite all occurrences of `torch.nn.LayerNorm` to `torch.nn.functional.layer_norm`."""

    @classmethod
    def rewrite(cls, node: Node) -> dict[Node, Node]:
        graph = node.graph
        if not (
            (graph_module := graph.owning_module) is not None
            and isinstance(layernorm := get_target_module(node), torch.nn.LayerNorm)
        ):
            return {}

        try:
            input_node = node.all_input_nodes[0]
        except IndexError:
            log.warning(f"LayerNorm node {node.name} has no input node: {node.format_node()}")
            return {}

        graph_module.register_parameter(weight_name := f"{node.target}_weight", layernorm.weight)
        graph_module.register_parameter(bias_name := f"{node.target}_bias", layernorm.bias)
        # pylint: disable=protected-access
        _ = graph_module._parameters.pop(f"{node.target}.weight", None)
        _ = graph_module._parameters.pop(f"{node.target}.bias", None)

        with graph.inserting_before(node):
            normalized_shape = list(layernorm.normalized_shape)
            weight_node = graph.get_attr(weight_name)
            bias_node = graph.get_attr(bias_name)
            layer_norm_node = graph.call_function(
                F.layer_norm,
                args=(input_node, normalized_shape, weight_node, bias_node),
                kwargs={"eps": layernorm.eps},
            )
        return {node: layer_norm_node}

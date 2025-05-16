from collections.abc import Callable

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd

from ...core.logger import log
from ...nn.modules.fake_quantizer import FakeQuantizer, PerTensorMixin
from ...nn.modules.qconv import QConv1d, QConv2d, QConv3d
from ...nn.modules.qlinear import QLinear
from ...options import Channel
from .node import get_target_module

BNFusionFunction = Callable[[_ConvNd, _BatchNorm], _ConvNd] | Callable[[torch.nn.Linear, _BatchNorm], torch.nn.Linear]


def replace_call_module_node_target(node: Node, new_module: Module) -> None:
    """Replace the module into new_module.

    Args:
        node: A Node. It is used for get name of module
        new_module: A new module to replace with.
    """
    graph_module = node.graph.owning_module
    if graph_module is None or not (node.op == "call_module" and isinstance(node.target, str)):
        return
    # `add_submodule` method overwrites the existing module
    graph_module.add_submodule(node.target, new_module)


def rescale_step_size_with_batchnorm(
    quantizer: FakeQuantizer,
    batchnorm: _BatchNorm,
) -> FakeQuantizer | None:
    """Rescales the step size of the fake quantizer with the BatchNormNd and returns it."""
    if not (
        isinstance(quantizer, FakeQuantizer)
        and isinstance(batchnorm, (_BatchNorm))
        and batchnorm.running_var is not None
    ):
        log.debug_warning(
            f"Expected quantizer to be {FakeQuantizer} and batchnorm to be one of types "
            f"{torch.nn.BatchNorm1d}, {torch.nn.BatchNorm2d}, {torch.nn.BatchNorm3d}, "
            f"with `running_var` but got {batchnorm} (running_var={batchnorm.running_var})"
        )
        return None

    scale = torch.rsqrt(batchnorm.running_var.data + batchnorm.eps)
    if batchnorm.weight is not None:
        scale = batchnorm.weight.data * scale

    if isinstance(quantizer, PerTensorMixin):
        log.warning(
            "Trying to BatchNorm Fuse a module that weight quantization is per-tensor. "
            "Automatically changed it to per-channel quantization"
        )
        quantizer = quantizer.as_per_channel(Channel(axis=0, size=batchnorm.num_features))  # type: ignore[assignment]
    quantizer.step_size.data = quantizer.step_size.data * scale.abs()
    return quantizer


def fuse_by_patterns(
    model: GraphModule, patterns: list[tuple[type[Module], type[Module]]], fusion_func: Callable
) -> None:
    """Fuses module/BN layers for inference purposes."""
    new_graph = model.graph
    for pattern in patterns:
        for node in new_graph.nodes:
            _fuse_by_pattern(model, node, pattern, fusion_func)
    new_graph.lint()
    model.recompile()


def _fuse_by_pattern(
    model: GraphModule, node: Node, pattern: tuple[type[Module], type[Module]], fusion_func: Callable
) -> None:
    if not isinstance(node.target, str) or not matches_module_pattern(pattern, node):
        return
    parent = node.all_input_nodes[0]
    if len(parent.users) > 1:  # Output of module is used by other nodes
        return
    if not (
        isinstance(module := get_target_module(parent), pattern[0])
        and isinstance(batchnorm := get_target_module(node), pattern[-1])
    ):
        raise TypeError("Not the type expected by the pattern")
    if (fused_module := fusion_func(module, batchnorm)) is None:
        return
    replace_call_module_node_target(parent, fused_module)
    node.replace_all_uses_with(parent)
    model.graph.erase_node(node)
    model.delete_submodule(node.target)


def fuse_conv_bn_eval(conv: _ConvNd, bn: _BatchNorm | None, rescale_step_size: bool = True) -> _ConvNd:
    """Fuse a convolution module and a batch normalization module in evaluation mode.

    If the convolution module is a quantized convolution and `rescale_step_size` is True,
    the step size of the weight quantizer is rescaled to reflect the batch normalization.

    Args:
        conv (`torch.nn.module.conv._ConvNd`): The convolution module to fuse.
        bn (`torch.nn._BatchNorm` | `None`): The batch normalization module to fuse.
        rescale_step_size (`bool`, optional): Whether to rescale the step size of the weight quantizer.
            Defaults to True.

    Returns:
        _ConvNd: The fused convolution module.
    """
    if bn is None:
        return conv
    fused_conv = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
    if rescale_step_size and isinstance(fused_conv, QConv1d | QConv2d | QConv3d) and fused_conv.weight_quantizer:
        fused_conv.weight_quantizer = rescale_step_size_with_batchnorm(fused_conv.weight_quantizer, bn)
    return fused_conv


def fuse_linear_bn_eval(
    linear: torch.nn.Linear, bn: _BatchNorm | None, rescale_step_size: bool = True
) -> torch.nn.Linear:
    """Fuse a linear module and a batch normalization module in evaluation mode.

    If the linear module is a quantized linear module and `rescale_step_size` is True,
    rescale the step size of the weight quantizer to reflect the batch normalization.

    Args:
        linear (`torch.nn.Linear`): The linear module to fuse.
        bn (`torch.nn.BatchNorm1d` | `None`): The batch normalization module to fuse.
        rescale_step_size (`bool`, optional): Whether to rescale the step size of the weight quantizer.
            Defaults to True.

    Returns:
        torch.nn.Linear: The fused linear module.
    """
    if bn is None:
        return linear
    fused_linear = torch.nn.utils.fusion.fuse_linear_bn_eval(linear, bn)
    if rescale_step_size and isinstance(fused_linear, QLinear) and fused_linear.weight_quantizer:
        rescale_step_size_with_batchnorm(fused_linear.weight_quantizer, bn)
    return fused_linear


def matches_module_pattern(pattern: tuple[type[Module], type[Module]], node: Node) -> bool:
    """Check current node matches with one of patterns.

    Returns:
        `True` if there is a pattern matched, `False` otherwise.
    """
    if not node.all_input_nodes:
        return False
    nodes = (node.all_input_nodes[0], node)
    return all(
        isinstance(get_target_module(current_node), expected_type)
        for expected_type, current_node in zip(pattern, nodes)
    )

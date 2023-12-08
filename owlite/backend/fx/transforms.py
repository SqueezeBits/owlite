"""Transformations for torch.fx.GraphModule"""
from collections.abc import Iterable
from itertools import combinations
from typing import Any, Callable, Optional, Union

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

from ...logger import log
from ...nn import modules as qnn
from ...nn.fake_quantizer import FakeQuantizer
from ...nn.modules.qmodule_mixins import UnaryNeuralQModuleMixin
from ..utils import get_most_common_device, nodestr
from .node import find_placeholders, get_target_module

GraphModuleTransform = Callable[[GraphModule], GraphModule]
GRAPH_MODULE_TRANSFORMS: dict[str, GraphModuleTransform] = {}


def apply_graph_module_transforms(graph_module: GraphModule) -> GraphModule:
    """Applies all registered graph module transforms

    Args:
        graph_module (GraphModule): a graph module

    Returns:
        GraphModule: the transformed graph module
    """
    for name, transform in GRAPH_MODULE_TRANSFORMS.items():
        log.debug(f"Applying graph module transform: {name}")
        graph_module = transform(graph_module)
    graph_module.recompile()
    return graph_module


def register_graph_module_transform(
    transform: GraphModuleTransform,
) -> GraphModuleTransform:
    """Registers a graph module transform globally. Note that the registration order matters.

    Use this function as a decorator to register your custom graph module transform. For example:
    @register_graph_module_transform
    def do_something_on_graph_module(graph_module: GraphModule) -> GraphModule:
        ...
    """
    name = transform.__name__
    if name in GRAPH_MODULE_TRANSFORMS:
        log.debug_warning(f"Overwriting existing GraphModule transform: {name}")
    GRAPH_MODULE_TRANSFORMS[name] = transform
    return transform


@register_graph_module_transform
def fix_input_parameter_names(graph_module: GraphModule) -> GraphModule:
    """Make the names of parameters of graph_module's forward method same as the original module.
    Note that this transform does nothing when torch<2.1.0

    Args:
        graph_module (GraphModule): the input graph module

    Returns:
        GraphModule: graph module with inputs renamed
    """
    for node in find_placeholders(graph_module.graph):
        if node.target.startswith("L_kwargs_") and node.target.endswith("_"):
            node.target = node.target[9:-1]
        elif node.target.startswith("L_") and node.target.endswith("_"):
            node.target = node.target[2:-1]
    return graph_module


@register_graph_module_transform
def fix_hard_coded_device(graph_module: GraphModule) -> GraphModule:
    """Fix hard coded devices to enanble data parallel.

    Args:
        graph_module (GraphModule): the input graph module

    Returns:
        GraphModule: graph module with inputs renamed
    """
    canary_tensor = torch.tensor(12.03, dtype=torch.float32, device=get_most_common_device(graph_module))
    graph_module.register_buffer("sqzb_module_device_canary", canary_tensor)

    graph = graph_module.graph
    with graph.inserting_before(next(iter(graph_module.graph.nodes))):
        canary = graph.get_attr("sqzb_module_device_canary")
        canary_device = graph.call_function(getattr, (canary, "device"))

    for node in graph.nodes:
        if node.kwargs.get("device", None) is not None:
            kwargs = node.kwargs.copy()
            kwargs["device"] = canary_device
            node.kwargs = kwargs

        if node.op == "call_method" and node.target == "to" and len(node.args) == 2 and isinstance(node.args[1], str):
            args = (node.args[0], canary_device)
            node.args = args

    return graph_module


def matches_module_pattern(pattern: Iterable[type], node: Node, modules: dict[str, Any]):
    """Check current node matches with one of patterns.

    Returns:
        True if there is a pattern matched. Else, False.
    """
    if len(node.args) == 0:
        return False
    nodes = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, Node):
            return False
        if current_node.op != "call_module":
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if not isinstance(modules[current_node.target], expected_type):
            # if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def replace_node_module(node: Node, module: dict[str, Any], new_module: torch.nn.Module):
    """Replacing module into new_module.

    Args:
        node: A Node. It is used for get name of module
        module: A old module.
        new_module: A new module to replace with.
    """
    if not isinstance(node.target, str):
        raise RuntimeError(f"Unable to get module name from node {nodestr(node)}")
    *parent, name = node.target.rsplit(".", 1)
    parent_name = parent[0] if parent else ""
    module[node.target] = new_module
    setattr(module[parent_name], name, new_module)


def _rescale_step_size_with_batchnorm(
    quantizer: FakeQuantizer,
    batchnorm: Union[torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d],
):
    if not isinstance(quantizer, FakeQuantizer):
        raise TypeError(
            "_rescale_step_size_with_batchnorm(): argument 'quantizer' (position 0) must be FakeQuantizer, "
            f"but found element of type {type(quantizer)} at pos 0"
        )
    if not isinstance(batchnorm, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        raise TypeError(
            "_rescale_step_size_with_batchnorm(): argument 'batchnorm' (position 0) must be BatchNormNd, "
            f"but found element of type {type(batchnorm)} at pos 0"
        )
    scale = batchnorm.weight.data * torch.rsqrt(batchnorm.running_var.data + batchnorm.eps)

    quantizer.step_size.data = quantizer.step_size.data * scale.abs()
    if not quantizer.per_channel.item():
        log.warning(
            "Trying to BatchNorm Fuse a module that weight quantization is per-tensor. "
            "Automatically changed it to per-channel quantization"
        )
        quantizer.per_channel.data = torch.tensor(True)
        quantizer.zero_point.data = quantizer.zero_point.data.broadcast_to(quantizer.step_size.shape)


def _fuse_by_patterns(
    model: GraphModule,
    patterns: list[tuple[type[torch.nn.Module]], ...],
    fusion_func: Callable[[torch.nn.Module, torch.nn.Module], torch.nn.Module],
):
    """Fuses module/BN layers for inference purposes."""
    modules = dict(model.named_modules())
    new_graph = model.graph
    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of module is used by other nodes
                    continue
                module = modules[node.args[0].target]
                batchnorm = modules[node.target]
                fused_module = fusion_func(module, batchnorm)
                if isinstance(module, UnaryNeuralQModuleMixin):
                    _rescale_step_size_with_batchnorm(fused_module.weight_quantizer, batchnorm)
                replace_node_module(node.args[0], modules, fused_module)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
                model.delete_submodule(node.target)
    new_graph.lint()
    model.recompile()
    return model


def fuse_linear_bn_with_quantized_bias(model: GraphModule):
    """Perform batchnorm fusing of owlite.nn.QLinear to quantize bias or hidden inputs.

    If a QLinear in the [owlite.nn.QLinear, torch.nn.BatchNorm1d] pattern quantizes a bias or hidden input,
    fuse it and adapt the step size of the QLinear weight quantizer. It is recommended that this process be
    performed before the quantizer's step_size is known, as quantizing bias or hidden inputs adds to the bias
    during batchnorm fusing and does not guarantee functionality.

    Args:
        model (GraphModule): a graph module
    """
    # in linear-bn pattern if bias or hidden input is quantized then fuse the bn.
    fused_list = []
    for node in model.graph.nodes:
        if not (node.op == "call_module" and isinstance(get_target_module(node), qnn.QLinear)):
            continue

        qlinear: qnn.QLinear = get_target_module(node)
        users = list(node.users)
        if (
            len(users) == 1
            and users[0].op == "call_module"
            and isinstance(get_target_module(users[0]), torch.nn.BatchNorm1d)
            and (qlinear.hidden_input_quantizer is not None or qlinear.bias_quantizer is not None)
        ):
            batchnorm_module: torch.nn.BatchNorm1d = get_target_module(users[0])
            fused_module = fuse_linear_bn_eval(qlinear, batchnorm_module)
            _rescale_step_size_with_batchnorm(qlinear.weight_quantizer, batchnorm_module)
            replace_node_module(
                node,
                dict(node.graph.owning_module.named_modules()),
                fused_module,
            )
            users[0].replace_all_uses_with(node)
            node.graph.erase_node(users[0])
            node.graph.owning_module.delete_submodule(users[0].target)
            node.graph.lint()
            node.graph.owning_module.recompile()
            fused_list.append(node)
    if len(fused_list) > 0:
        log.warning(
            "Linear-BatchNorm patterns have been detected when quantizing bias or hidden input in torch.nn.Linear. "
            "Automatically fusing the Linear-BatchNorm patterns"
        )
        log.warning(f"Fused node : {fused_list}")


def fuse_bn(model: GraphModule):
    """Fuse Conv-BatchNorm patterns in model into Conv

    Args:
        model (GraphModule): a graph module, possibly wrapped by dp or ddp.
    """
    conv_patterns = [
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d),
        (qnn.QConv1d, torch.nn.BatchNorm1d),
        (qnn.QConv2d, torch.nn.BatchNorm2d),
        (qnn.QConv3d, torch.nn.BatchNorm3d),
    ]
    linear_patterns = [
        (torch.nn.Linear, torch.nn.BatchNorm1d),
        (qnn.QLinear, torch.nn.BatchNorm1d),
    ]
    training_status = model.training
    model.eval()
    _fuse_by_patterns(model, conv_patterns, fuse_conv_bn_eval)
    _fuse_by_patterns(model, linear_patterns, fuse_linear_bn_eval)
    model.train(training_status)


def fold_zp_to_bias(qmodel: torch.nn.Module):
    """folding all zeropoints of asymmetric quantization to bias of following operations

    Args:
        qmodel: model to fold
    """
    for _, module in qmodel.named_modules():
        if isinstance(module, (qnn.QConv2d, qnn.QLinear)):
            module.fold_input_quantizer_zero_point_to_bias()


def unfold_zp_to_bias(qmodel):
    """folding zeropoint of asymmetric quantization from bias of following operation

    Args:
        qmodel: model to unfold
    """
    for _, module in qmodel.named_modules():
        if isinstance(module, (qnn.QConv2d, qnn.QLinear)):
            module.unfold_input_quantizer_zero_point_to_bias()


def fuse_redundant_quantizers(graph_module: GraphModule):
    """Fuses all redundant quantizers

    Args:
        graph_module (GraphModule): a graph module
    """
    for node in graph_module.graph.nodes:
        fuse_redundant_input_quantizers_of(node)


def fuse_redundant_input_quantizers_of(node: Node):
    """Merge quantizers with src_node as argument that have the same quant_scheme

    Args:
        node: the node to fuse redundant input quantizers
    """
    graph_module = node.graph.owning_module
    if graph_module is None:
        return
    quantizer_nodes: list[Optional[Node]] = [
        *filter(
            lambda n: n.op == "call_module" and isinstance(get_target_module(n), FakeQuantizer),
            node.users,
        )
    ]

    for i, j in combinations(range(len(quantizer_nodes)), 2):
        node_to_reuse = quantizer_nodes[i]
        node_to_remove = quantizer_nodes[j]
        if not (node_to_reuse is not None and node_to_remove is not None):
            continue
        quantizer_to_reuse = get_target_module(node_to_reuse)
        quantizer_to_remove = get_target_module(node_to_remove)
        if not (
            isinstance(quantizer_to_reuse, FakeQuantizer)
            and isinstance(quantizer_to_remove, FakeQuantizer)
            and quantizer_to_reuse.options == quantizer_to_remove.options
        ):
            continue

        for child_node in node_to_remove.users:
            child_module = get_target_module(child_node)
            if not (isinstance(child_module, UnaryNeuralQModuleMixin) and child_module.input_quantizer is not None):
                continue
            child_module.input_quantizer = quantizer_to_reuse
        node_to_remove.replace_all_uses_with(node_to_reuse)
        graph_module.graph.erase_node(node_to_remove)
        graph_module.delete_submodule(node_to_remove.target)
        quantizer_nodes[j] = None
        graph_module.recompile()


def clip_narrow_range_weights(graph_module: GraphModule):
    """Clips weights with a narrow range of QConv in the graph_module.

    Args:
        graph_module (GraphModule): a graph module
    """
    for _, module in graph_module.named_modules(remove_duplicate=True):
        if isinstance(module, (UnaryNeuralQModuleMixin)) and module.weight_quantizer.narrow:
            module.clip_weight()

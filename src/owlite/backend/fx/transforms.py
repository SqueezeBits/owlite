import inspect
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm

from ...nn.modules import (
    QConv1d,
    QConv2d,
    QConv3d,
    QConvBn1d,
    QConvBn2d,
    QConvBn3d,
    QLinear,
    UnaryNeuralQModuleMixin,
)
from ...owlite_core.logger import log
from ..signature import Signature
from ..utils import get_most_common_device, nodestr, normalize_parameter_name
from .batchnorm_fusion_helper import (
    fuse_by_patterns,
    fuse_conv_bn_eval,
    fuse_linear_bn_eval,
    replace_call_module_node_target,
)
from .node import find_placeholders, get_target_module

GraphModuleTransform = (
    Callable[[GraphModule], GraphModule] | Callable[[GraphModule, tuple[Any, ...], dict[str, Any]], GraphModule]
)
GRAPH_MODULE_TRANSFORMS: dict[str, GraphModuleTransform] = {}


def apply_graph_module_transforms(
    graph_module: GraphModule,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> GraphModule:
    """Apply all registered graph module transforms.

    Args:
        graph_module (GraphModule): a graph module
        args (tuple[Any, ...]): arguments to be provided for the module's forward method
        kwargs (dict[str, Any]): keyword arguments to be provided for the module's forward method

    Returns:
        GraphModule: the transformed graph module
    """
    for name, transform in GRAPH_MODULE_TRANSFORMS.items():
        log.debug(f"Applying graph module transform: {name}")
        graph_module = (
            transform(graph_module)  # type: ignore[call-arg]
            if len(inspect.signature(transform).parameters) == 1
            else transform(graph_module, args, kwargs)  # type: ignore[call-arg]
        )
    graph_module.recompile()
    return graph_module


def register_graph_module_transform(
    transform: GraphModuleTransform,
) -> GraphModuleTransform:
    """Register a graph module transform globally. Note that the registration order matters.

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

    Note that this transform does nothing when torch<2.1.0.

    Args:
        graph_module (GraphModule): the input graph module

    Returns:
        GraphModule: graph module with inputs renamed
    """
    for node in find_placeholders(graph_module.graph):
        if not isinstance(node.target, str):
            continue
        target = normalize_parameter_name(node.target)
        if node.target != target:
            log.debug(f"Renaming placeholder {node.target} -> {target}")
            node.target = target
    return graph_module


@register_graph_module_transform
def fix_forward_argument_ordering(
    graph_module: GraphModule,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> GraphModule:
    """Reorder graph module input arguments to meet the ordering in original module.

    Args:
        graph_module (GraphModule): the input graph module
        args (tuple[Any, ...]): arguments to be provided for the module's forward method
        kwargs (dict[str, Any]): keyword arguments to be provided for the module's forward method

    Returns:
        GraphModule: graph module with inputs reordered
    """
    graph = graph_module.graph
    names = list(Signature.from_module(graph_module, args, kwargs).keys())
    log.debug(f"Names from signature: {names}")

    placeholders = find_placeholders(graph)
    log.debug(f"Original placeholders: {[nodestr(p) for p in placeholders]}")

    def get_index(node: Node) -> int:
        if isinstance((target := node.target), str) and target in names:
            return names.index(target)
        return len(names)

    placeholders = [*sorted(placeholders, key=get_index, reverse=True)]
    log.debug(f"Reverse-sorted placeholders: {[nodestr(p) for p in placeholders]}")

    for placeholder in placeholders:
        with graph.inserting_before():
            reordered_placeholder = graph.placeholder(f"{placeholder.name}_reordered")
            reordered_placeholder.target = placeholder.target
            placeholder.replace_all_uses_with(reordered_placeholder)
            graph.erase_node(placeholder)

    return graph_module


@register_graph_module_transform
def fix_hard_coded_device(graph_module: GraphModule) -> GraphModule:
    """Fix hard coded devices to enable data parallel.

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
        graph_module.meta["canary_device_node"] = canary_device

    for node in graph.nodes:
        if node.kwargs.get("device", None) is not None:
            kwargs = node.kwargs.copy()
            kwargs["device"] = canary_device
            node.kwargs = kwargs

        if node.op == "call_method" and node.target == "to" and len(node.args) == 2 and isinstance(node.args[1], str):
            args = (node.args[0], canary_device)
            node.args = args

    return graph_module


@register_graph_module_transform
def canonicalize_silu(graph_module: GraphModule) -> GraphModule:
    """Decompose all occurences of `torch.nn.SiLU` and `torch.nn.functional.silu` by sigmoid and mul node pairs.

    Args:
        graph_module (GraphModule): the input graph module

    Returns:
        GraphModule: the graph module all of whose silu nodes are canonicalized.
    """
    graph = graph_module.graph
    for node in graph.nodes:
        if not isinstance(node, Node):
            continue
        module = get_target_module(node)
        if isinstance(module, torch.nn.SiLU) or (
            node.op == "call_function" and node.target is torch.nn.functional.silu
        ):
            input_node = node.all_input_nodes[0]
            with graph.inserting_before(node):
                sigmoid_node = graph.call_function(torch.nn.functional.sigmoid, args=(input_node,))
                mul_node = graph.call_function(torch.mul, args=(input_node, sigmoid_node))
            node.replace_all_uses_with(mul_node)
    graph.lint()
    return graph_module


@register_graph_module_transform
def canonicalize_hstack(graph_module: GraphModule) -> GraphModule:
    """Replace call_function(torch.hstack) by call_function(torch.cat)."""
    graph = graph_module.graph
    for node in graph.nodes:
        if not (
            isinstance(node, Node)
            and node.op == "call_function"
            and node.target is torch.hstack
            and isinstance((tensors := node.args[0] if node.args else node.kwargs.get("tensors", None)), list | tuple)
        ):
            continue
        with graph.inserting_before(node):
            cat_node = graph.call_function(torch.cat, args=(tensors,), kwargs={"dim": 1})
        node.replace_all_uses_with(cat_node)
    graph.lint()
    return graph_module


def fuse_linear_bn_with_quantized_bias(model: GraphModule) -> None:
    """Perform batchnorm fusing of `owlite.nn.QLinear` to quantize bias or hidden inputs.

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
        if not isinstance((qlinear := get_target_module(node)), QLinear):
            continue

        users = list(node.users)
        if (
            len(users) == 1
            and isinstance((batchnorm_module := get_target_module(users[0])), torch.nn.BatchNorm1d)
            and qlinear.weight_quantizer is not None
            and (qlinear.hidden_input_quantizer is not None or qlinear.bias_quantizer is not None)
        ):
            if (
                not isinstance(fused_module := fuse_linear_bn_eval(qlinear, batchnorm_module), QLinear)
                or fused_module.weight_quantizer is None
            ):
                log.debug_warning(
                    "fuse_linear_bn_eval() returns invalid type."
                    f" - got {type(fused_module)}, but expected: {QLinear}"
                )
                return
            replace_call_module_node_target(node, fused_module)
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


def fuse_bn(model: GraphModule) -> None:
    """Fuse Conv-BatchNorm or Linear-BatchNorm patterns in model.

    Args:
        model (GraphModule): a graph module, possibly wrapped by dp or ddp.
    """
    training_status = model.training
    model.eval()
    for module in model.modules():
        if isinstance(module, QConvBn1d | QConvBn2d | QConvBn3d) and isinstance(
            fused_module := fuse_conv_bn_eval(module.qconv, module.bn), type(module.qconv)
        ):
            module.qconv = fused_module  # type: ignore[assignment]
            module.bn = None

    conv_patterns: list[tuple[type[Module], type[Module]]] = [
        (QConv1d, torch.nn.BatchNorm1d),
        (QConv2d, torch.nn.BatchNorm2d),
        (QConv3d, torch.nn.BatchNorm3d),
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d),
    ]
    linear_patterns: list[tuple[type[Module], type[Module]]] = [
        (torch.nn.Linear, torch.nn.BatchNorm1d),
        (QLinear, torch.nn.BatchNorm1d),
    ]
    fuse_by_patterns(model, conv_patterns, fuse_conv_bn_eval)
    fuse_by_patterns(model, linear_patterns, fuse_linear_bn_eval)
    model.train(training_status)


def clip_narrow_range_weights(graph_module: GraphModule) -> None:
    """Clip weights with a narrow range of quantized modules in the graph_module.

    Args:
        graph_module (GraphModule): a graph module
    """
    for _, module in graph_module.named_modules(remove_duplicate=True):
        if (
            isinstance(module, (UnaryNeuralQModuleMixin))
            and (weight_quantizer := module.weight_quantizer) is not None
            and weight_quantizer.narrow_range
        ):
            module.clip_weight()


def qconv_bn_to_qconvbn(model: GraphModule) -> None:
    """Convert the QConvNd and BatchNormNd patterns in the model to QConvBnNd.

    Args:
        model(GraphModule): Model with modules to convert
    """

    def merge_qconv_bn(qconv: QConv1d | QConv2d | QConv3d, bn: _BatchNorm) -> QConvBn1d | QConvBn2d | QConvBn3d | None:
        if not qconv.int32_bias:
            return None
        if isinstance(qconv, QConv1d) and isinstance(bn, torch.nn.BatchNorm1d):
            return QConvBn1d(qconv, bn)
        if isinstance(qconv, QConv2d) and isinstance(bn, torch.nn.BatchNorm2d):
            return QConvBn2d(qconv, bn)
        if isinstance(qconv, QConv3d) and isinstance(bn, torch.nn.BatchNorm3d):
            return QConvBn3d(qconv, bn)
        return None

    qconv_bn_patterns: list[tuple[type[Module], type[Module]]] = [
        (QConv1d, torch.nn.BatchNorm1d),
        (QConv2d, torch.nn.BatchNorm2d),
        (QConv3d, torch.nn.BatchNorm3d),
    ]
    fuse_by_patterns(model, qconv_bn_patterns, merge_qconv_bn)

import inspect
from typing import Any, Callable, Optional, Union

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.nn.modules.conv import _ConvNd
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

from ...nn import FakePerTensorQuantizer, FakeQuantizer
from ...nn import modules as qnn
from ...nn.modules.qmodule_mixins import UnaryNeuralQModuleMixin
from ...options import Channel
from ...owlite_core.logger import log
from ..signature import Signature
from ..utils import get_most_common_device, nodestr, normalize_parameter_name
from .node import find_placeholders, get_target_module

GraphModuleTransform = Union[
    Callable[[GraphModule], GraphModule], Callable[[GraphModule, tuple[Any, ...], dict[str, Any]], GraphModule]
]
GRAPH_MODULE_TRANSFORMS: dict[str, GraphModuleTransform] = {}


def apply_graph_module_transforms(
    graph_module: GraphModule,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> GraphModule:
    """Applies all registered graph module transforms

    Args:
        graph_module (GraphModule): a graph module

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

    Returns:
        GraphModule: graph module with inputs reordered
    """
    graph = graph_module.graph
    names = [name for name, _ in Signature.from_module(graph_module, args, kwargs)]
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
    """Decomposes all appearances of `torch.nn.SiLU` and `torch.nn.functional.silu`
    by two separate sigmoid and mul nodes.

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
    """Replaces call_function(torch.hstack) by call_function(torch.cat)"""
    graph = graph_module.graph
    for node in graph.nodes:
        if not (
            isinstance(node, Node)
            and node.op == "call_function"
            and node.target is torch.hstack
            and isinstance((tensors := node.args[0] if node.args else node.kwargs.get("tensors", None)), (list, tuple))
        ):
            continue
        with graph.inserting_before(node):
            cat_node = graph.call_function(torch.cat, args=(tensors,), kwargs={"dim": 1})
        node.replace_all_uses_with(cat_node)
    graph.lint()
    return graph_module


def matches_module_pattern(pattern: tuple[type[torch.nn.Module], type[torch.nn.Module]], node: Node) -> bool:
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


def replace_call_module_node_target(node: Node, new_module: torch.nn.Module) -> None:
    """Replacing module into new_module.

    Args:
        node: A Node. It is used for get name of module
        new_module: A new module to replace with.
    """
    graph_module = node.graph.owning_module
    if graph_module is None or not (node.op == "call_module" and isinstance(node.target, str)):
        return
    # `add_submodule` method overwrites the existing module
    graph_module.add_submodule(node.target, new_module)


def _rescale_step_size_with_batchnorm(
    quantizer: FakeQuantizer,
    batchnorm: Union[torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d],
) -> Optional[FakeQuantizer]:
    """Rescales the step size of the fake quantizer with the BatchNormNd and returns it."""
    if not (
        isinstance(quantizer, FakeQuantizer)
        and isinstance(batchnorm, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
        and batchnorm.running_var is not None
    ):
        log.debug_warning(
            f"Expected quantizer to be {FakeQuantizer} and batchnorm to be one of types "
            f"{torch.nn.BatchNorm1d}, {torch.nn.BatchNorm2d}, {torch.nn.BatchNorm3d}, "
            f"with `running_var` but got {batchnorm} (running_var={batchnorm.running_var})"
        )
        return None
    scale = batchnorm.weight.data * torch.rsqrt(batchnorm.running_var.data + batchnorm.eps)

    if isinstance(quantizer, FakePerTensorQuantizer):
        log.warning(
            "Trying to BatchNorm Fuse a module that weight quantization is per-tensor. "
            "Automatically changed it to per-channel quantization"
        )
        quantizer.as_per_channel(Channel(axis=0, size=batchnorm.num_features))
    quantizer.step_size.data = quantizer.step_size.data * scale.abs()
    return quantizer


BatchNorm = Union[torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d]

FusionFunction = Union[
    Callable[[_ConvNd, BatchNorm], torch.nn.Module],
    Callable[[torch.nn.Linear, BatchNorm], torch.nn.Module],
]


def _fuse_by_patterns(
    model: GraphModule,
    patterns: list[tuple[type[torch.nn.Module], type[torch.nn.Module]]],
    fusion_func: FusionFunction,
) -> None:
    """Fuses module/BN layers for inference purposes."""
    new_graph = model.graph
    for pattern in patterns:
        node: Node
        for node in new_graph.nodes:
            if not isinstance(node.target, str):
                continue
            if not matches_module_pattern(pattern, node):
                continue
            parent = node.all_input_nodes[0]
            if len(parent.users) > 1:  # Output of module is used by other nodes
                continue
            if not (
                isinstance((module := get_target_module(parent)), (_ConvNd, torch.nn.Linear))
                and isinstance(
                    (batchnorm := get_target_module(node)),
                    (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
                )
            ):
                continue
            fused_module = fusion_func(module, batchnorm)  # type: ignore[arg-type]
            if isinstance(fused_module, UnaryNeuralQModuleMixin) and fused_module.weight_quantizer:
                fused_module.weight_quantizer = _rescale_step_size_with_batchnorm(
                    fused_module.weight_quantizer, batchnorm
                )
            replace_call_module_node_target(parent, fused_module)
            node.replace_all_uses_with(parent)
            new_graph.erase_node(node)
            model.delete_submodule(node.target)
    new_graph.lint()
    model.recompile()


def fuse_linear_bn_with_quantized_bias(model: GraphModule) -> None:
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
        if not isinstance((qlinear := get_target_module(node)), qnn.QLinear):
            continue

        users = list(node.users)
        if (
            len(users) == 1
            and isinstance(
                (batchnorm_module := get_target_module(users[0])),
                torch.nn.BatchNorm1d,
            )
            and qlinear.weight_quantizer is not None
            and (qlinear.hidden_input_quantizer is not None or qlinear.bias_quantizer is not None)
        ):
            if (
                not isinstance(fused_module := fuse_linear_bn_eval(qlinear, batchnorm_module), qnn.QLinear)
                or fused_module.weight_quantizer is None
            ):
                log.debug_warning(
                    "fuse_linear_bn_eval() returns invalid type."
                    f" - got {type(fused_module)}, but expected: {qnn.QLinear}"
                )
                return
            fused_module.weight_quantizer = _rescale_step_size_with_batchnorm(
                fused_module.weight_quantizer, batchnorm_module
            )
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
    """Fuse Conv-BatchNorm patterns in model into Conv

    Args:
        model (GraphModule): a graph module, possibly wrapped by dp or ddp.
    """
    conv_patterns: list[tuple[type[torch.nn.Module], type[torch.nn.Module]]] = [
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d),
        (qnn.QConv1d, torch.nn.BatchNorm1d),
        (qnn.QConv2d, torch.nn.BatchNorm2d),
        (qnn.QConv3d, torch.nn.BatchNorm3d),
    ]
    linear_patterns: list[tuple[type[torch.nn.Module], type[torch.nn.Module]]] = [
        (torch.nn.Linear, torch.nn.BatchNorm1d),
        (qnn.QLinear, torch.nn.BatchNorm1d),
    ]
    training_status = model.training
    model.eval()
    _fuse_by_patterns(model, conv_patterns, fuse_conv_bn_eval)
    _fuse_by_patterns(model, linear_patterns, fuse_linear_bn_eval)
    model.train(training_status)


def fold_zp_to_bias(qmodel: torch.nn.Module) -> None:
    """folding all zeropoints of asymmetric quantization to bias of following operations

    Args:
        qmodel: model to fold
    """
    for _, module in qmodel.named_modules():
        if isinstance(module, (qnn.QConv2d, qnn.QLinear)):
            module.fold_input_quantizer_zero_point_to_bias()


def unfold_zp_to_bias(qmodel: torch.nn.Module) -> None:
    """folding zeropoint of asymmetric quantization from bias of following operation

    Args:
        qmodel: model to unfold
    """
    for _, module in qmodel.named_modules():
        if isinstance(module, (qnn.QConv2d, qnn.QLinear)):
            module.unfold_input_quantizer_zero_point_to_bias()


def clip_narrow_range_weights(graph_module: GraphModule) -> None:
    """Clips weights with a narrow range of quantized modules in the graph_module.

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

from collections.abc import Callable

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
from ...nn.modules.granularity_mixin import PerTensorMixin
from ...owlite_core.logger import log
from ..utils import get_most_common_device
from .batchnorm_fusion_helper import (
    fuse_by_patterns,
    fuse_conv_bn_eval,
    fuse_linear_bn_eval,
    replace_call_module_node_target,
)
from .node import get_target_module

GraphModuleTransform = Callable[[GraphModule], GraphModule]
GRAPH_MODULE_TRANSFORMS: dict[str, GraphModuleTransform] = {}


def apply_graph_module_transforms(
    graph_module: GraphModule,
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
        graph_module = transform(graph_module)
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


def fuse_bn_into_qlinear_with_quantized_bias(model: GraphModule) -> None:
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


def fuse_bn_into_qmodule_with_per_tensor_quantizer(model: GraphModule) -> None:
    """Fuse a quantized module and a batch normalization module when the module uses per-tensor quantization.

    Args:
        model (GraphModule): a graph module.
    """
    qmodule_bn_patterns: list[tuple[type[Module], type[Module]]] = [
        (QConv1d, torch.nn.BatchNorm1d),
        (QConv2d, torch.nn.BatchNorm2d),
        (QConv3d, torch.nn.BatchNorm3d),
        (QLinear, torch.nn.BatchNorm1d),
    ]

    def _fuse_bn_into_qmodule_with_per_tensor_quantizer(
        module: UnaryNeuralQModuleMixin, bn: _BatchNorm | None
    ) -> Module | None:
        """Fuse a module and a batch normalization when the module uses per-tensor quantization in model.

        Args:
            module (`UnaryNeuralQModuleMixin`): The module to fuse.
            bn (`torch.nn.BatchNorm` | None): The batch normalization module to fuse.

        Returns:
            torch.nn.Module | None: The fused module or None if the module does not use per-tensor quantization.
        """
        if (weight_quantizer := module.weight_quantizer) is not None and isinstance(weight_quantizer, PerTensorMixin):
            if isinstance(module, QConv1d | QConv2d | QConv3d):
                log.debug(f"Fuse {module} and `BatchNormNd` with per-tensor weight quantizer({weight_quantizer.id})")
                return fuse_conv_bn_eval(module, bn, rescale_step_size=False)
            if isinstance(module, QLinear):
                log.debug(f"Fuse {module} and `BatchNorm1d` with per-tensor weight quantizer({weight_quantizer.id})")
                return fuse_linear_bn_eval(module, bn, rescale_step_size=False)
        return None

    training_status = model.training
    model.eval()
    fuse_by_patterns(model, qmodule_bn_patterns, _fuse_bn_into_qmodule_with_per_tensor_quantizer)
    model.train(training_status)


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


def qconv_bn_to_qconvbn_with_int32bias(model: GraphModule) -> None:
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

import torch
from torch.fx import GraphModule
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm

from ...core.logger import log
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
from .batchnorm_fusion_helper import (
    fuse_by_patterns,
    fuse_conv_bn_eval,
    fuse_linear_bn_eval,
    replace_call_module_node_target,
)
from .node import get_target_module


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

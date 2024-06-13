import torch

from .fake_quantizer import (
    FakeFPQuantizer,
    FakeINTQuantizer,
    FakePerChannelFPQuantizer,
    FakePerChannelINTQuantizer,
    FakePerTensorFPQuantizer,
    FakePerTensorINTQuantizer,
    FakeQuantizer,
)
from .qconv import QConv1d, QConv2d, QConv3d
from .qconvbn import QConvBn1d, QConvBn2d, QConvBn3d
from .qlinear import QLinear
from .qmodule_mixins import UnaryNeuralQModuleMixin


def promote_to_qmodule(
    cls: type[torch.nn.Module],
) -> type[QConv1d] | type[QConv2d] | type[QConv3d] | type[QLinear] | None:
    """Convert a torch.nn.Module subclass to its quantized counterpart if exists.

    Args:
        cls (type[torch.nn.Module]): a subclass of a torch.nn.Module

    Returns:
        type[QConv1d] | type[QConv2d] | type[QConv3d] | type[QLinear] | None: a quantized counterpart of the `cls` if
            exists. None otherwise.
    """
    quantized_class_map: dict[type[torch.nn.Module], type[QConv1d] | type[QConv2d] | type[QConv3d] | type[QLinear]] = {
        torch.nn.Conv1d: QConv1d,
        torch.nn.Conv2d: QConv2d,
        torch.nn.Conv3d: QConv3d,
        torch.nn.Linear: QLinear,
    }
    return quantized_class_map.get(cls, None)


def enable_quantizers(module: torch.nn.Module) -> None:
    """Enable all fake quantizers in the module.

    Args:
        module (torch.nn.Module): The module containing fake quantizers to enable or disable.
    """
    for _, submodule in module.named_modules():
        if isinstance(submodule, FakeQuantizer):
            submodule.enable()


def disable_quantizers(module: torch.nn.Module) -> None:
    """Disable all fake quantizers in the module.

    Args:
        module (torch.nn.Module): The module containing fake quantizers to enable or disable.
    """
    for _, submodule in module.named_modules():
        if isinstance(submodule, FakeQuantizer):
            submodule.disable()

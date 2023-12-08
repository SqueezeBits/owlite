"""owlite.quant.modules

Modules that utilize the quantizer
"""
from typing import Optional, Union

import torch

from .qconv import QConv1d, QConv2d, QConv3d
from .qlinear import QLinear
from .qmodule_mixins import UnaryNeuralQModuleMixin

QModule = Union[QConv1d, QConv2d, QConv3d, QLinear]


def promote_to_qmodule(cls: type[torch.nn.Module]) -> Optional[type[QModule]]:
    """Convert a torch.nn.Module subclass to its quantized counterpart if exists.

    Args:
        cls (type[torch.nn.Module]): a subclass of a torch.nn.Module

    Returns:
        Optional[type[QModule]]: a quantized counterpart of the `cls` if exists. None otherwise.
    """
    quantized_class_map: dict[type[torch.nn.Module], type[QModule]] = {
        torch.nn.Conv1d: QConv1d,
        torch.nn.Conv2d: QConv2d,
        torch.nn.Conv3d: QConv3d,
        torch.nn.Linear: QLinear,
    }
    return quantized_class_map.get(cls, None)

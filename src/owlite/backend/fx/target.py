# pylint: disable=missing-function-docstring
import operator

import torch
from torch.fx.node import Target as FXTarget

from ..utils import camel_to_snake
from .types import TorchTarget


def torch_targets(fn_name: str) -> list[TorchTarget]:
    """Finds all torch.* function targets with given fn_name

    Args:
        fn_name (str): torch function name to find (e.g. "randn" for targeting torch.randn)

    Returns:
        list[TorchTarget]: list of all torch function targets
    """
    inplace_fn_name = f"{fn_name}_"
    targets: list[TorchTarget] = []
    if hasattr(torch, fn_name):
        targets.append(getattr(torch, fn_name))
    if hasattr(torch, inplace_fn_name):
        targets.append(getattr(torch, inplace_fn_name))
    if hasattr(torch.Tensor, fn_name):
        targets.append(fn_name)
    if hasattr(torch.Tensor, inplace_fn_name):
        targets.append(inplace_fn_name)
    return targets


def functional_targets(op_name: str) -> list[FXTarget]:
    targets: list[FXTarget] = []
    if hasattr(torch.nn.functional, op_name):
        targets.append(getattr(torch.nn.functional, op_name))
    inplace_op_name = f"{op_name}_"
    if hasattr(torch.nn.functional, inplace_op_name):
        targets.append(getattr(torch.nn.functional, inplace_op_name))
    for i in (1, 2, 3):
        op_name_nd = f"{op_name}{i}d"
        if hasattr(torch.nn.functional, op_name_nd):
            targets.append(getattr(torch.nn.functional, op_name_nd))
    return targets


def nn_targets(module_name: str) -> list[FXTarget]:
    targets: list[FXTarget] = []
    if hasattr(torch.nn, module_name):
        targets.append(getattr(torch.nn, module_name))

    for n in (1, 2, 3):
        module_name_nd = f"{module_name}{n}d"
        if hasattr(torch.nn, module_name_nd):
            targets.append(getattr(torch.nn, module_name_nd))

    return targets


def builtin_targets(op_name: str) -> list[FXTarget]:
    inplace_op_name = f"i{op_name}"
    targets: list[FXTarget] = []
    if hasattr(operator, op_name):
        targets.append(getattr(operator, op_name))
    if hasattr(operator, inplace_op_name):
        targets.append(getattr(operator, inplace_op_name))
    return targets


def all_torch_functions(op_name: str) -> list[FXTarget]:
    return torch_targets(op_name) + functional_targets(op_name)


def all_torch_targets(op_name: str) -> list[FXTarget]:
    snake_op_name = camel_to_snake(op_name)
    return nn_targets(op_name) + all_torch_functions(snake_op_name)


def all_targets(op_name: str) -> list[FXTarget]:
    snake_op_name = camel_to_snake(op_name)
    return builtin_targets(snake_op_name) + all_torch_targets(snake_op_name)


ADD_TARGETS = (*all_targets("add"),)

SUB_TARGETS = (*all_targets("sub"), *all_torch_targets("subtract"))

MUL_TARGETS = (*all_targets("mul"), *all_torch_targets("multiply"))

DIV_TARGETS = (
    operator.truediv,
    operator.itruediv,
    operator.floordiv,
    operator.ifloordiv,
    *all_torch_targets("div"),
    *all_torch_targets("divide"),
    *all_torch_targets("floor_divide"),
    *all_torch_targets("true_divide"),
)

ARITHMETIC_TARGETS = (*ADD_TARGETS, *SUB_TARGETS, *MUL_TARGETS, *DIV_TARGETS)

CONSTANT_TARGETS = (
    *torch_targets("zero_"),
    *torch_targets("new_tensor"),
    *torch_targets("empty"),
    *torch_targets("empty_like"),
    *torch_targets("new_empty"),
    *torch_targets("empty_strided"),
    *torch_targets("zeros"),
    *torch_targets("zeros_like"),
    *torch_targets("new_zeros"),
    *torch_targets("ones"),
    *torch_targets("ones_like"),
    *torch_targets("new_ones"),
    *torch_targets("full"),
    *torch_targets("full_like"),
    *torch_targets("new_full"),
    *torch_targets("rand"),
    *torch_targets("randn"),
    *torch_targets("randint"),
    *torch_targets("rand_like"),
    *torch_targets("randn_like"),
    *torch_targets("randint_like"),
    *torch_targets("arange"),
    *torch_targets("as_tensor"),
    *torch_targets("asarray"),
    *torch_targets("bartlett_window"),
    *torch_targets("eye"),
    *torch_targets("from_file"),
    *torch_targets("from_numpy"),
    *torch_targets("hamming_window"),
    *torch_targets("hann_window"),
)

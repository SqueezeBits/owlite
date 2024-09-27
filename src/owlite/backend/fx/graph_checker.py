# pylint: disable=protected-access
# mypy: disable-error-code=attr-defined
import builtins
import inspect
import os
from collections.abc import Callable
from types import ModuleType
from typing import Any, TypeGuard

import _operator
import numpy
import torch
import torch._dynamo
from torch.fx.graph_module import GraphModule

from ...core.logger import log
from ..utils import targetstr
from .node import get_torch_target
from .types import TorchTarget


class UnsupportedFunctionCallError(Exception):
    """Exception indicating that an external library function call is detected in a graph module."""


class UnsupportedModuleCallError(Exception):
    """Exception indicating that an external library module call is detected in a graph module."""


class UnsupportedAutogradFunctionCallError(Exception):
    """Exception indicating that a custom autograd function call is detected in a graph module."""


def validate_procedure_calls(graph_module: GraphModule) -> None:
    """Assert all "call_function" nodes in the given graph module are targeting supported functions.

    A native function is either a function from `torch` or a python built-in function.

    Args:
        graph_module (GraphModule): a graph module.

    Raises:
        UnsupportedFunctionCallError: if any external library function call is detected.
    """
    if not (foreign_targets := collect_foreign_functions(graph_module)):
        return
    message = (
        "OwLite cannot process external library functions called inside the model's forward method. "
        "Please replace the following external library function(s) by native PyTorch functions or builtin functions:\n"
    )  # UX
    for module, targets in foreign_targets.items():
        targets_repr = ", ".join(f"'{target_repr(target)}'" for target in targets)
        module_repr = "unknown module(s)" if module is None else f"the module '{module.__name__}'"
        message += f"* {targets_repr} from {module_repr}\n"
    log.error(message)
    raise UnsupportedFunctionCallError("Found third party library function calls in the model implementation")  # UX


def collect_foreign_functions(graph_module: GraphModule) -> dict[ModuleType | None, set[Callable[..., Any]]]:
    """Collect all functions from external python modules called in the given graph module.

    Args:
        graph_module (GraphModule): a graph module.

    Returns:
        dict[ModuleType | None, set[Callable[..., Any]]]: a dictionary mapping each external python module
        (or `None` if the module is unidentified) to the function called in the graph module.
    """
    foreign_targets: dict[ModuleType | None, set[Callable[..., Any]]] = {}
    for node in graph_module.graph.nodes:
        if not (
            node.op in ("call_function", "call_module")
            and (target := get_torch_target(node)) is not None
            and is_unsupported_function(target, (module := get_module(target)))
        ):
            continue
        if module not in foreign_targets:
            foreign_targets[module] = set()
        foreign_targets[module].add(target)
    return foreign_targets


def target_repr(target: Callable[..., Any]) -> str:
    """Represent target as a human interpretable string, handling numpy functions wrapped by `torch._dynamo`.

    Args:
        target (Target): the target of an FX node whose op is "call_function"

    Returns:
        str: the target's representation.
    """

    def _normalize_tnp(name: str) -> str:
        return name.replace("torch._numpy", "numpy").replace("._funcs_impl", "").replace("._reductions_impl", "")

    if isinstance(target, torch._dynamo.utils.numpy_method_wrapper):
        return f"numpy.NDArray.{target.method}"
    if isinstance(target, torch._dynamo.utils.numpy_operator_wrapper):
        return _normalize_tnp(targetstr(target.op))
    if isinstance(target, torch._dynamo.utils.numpy_to_tensor_wrapper):
        return _normalize_tnp(targetstr(target.f))
    return targetstr(target)


def is_unsupported_function(target: TorchTarget, module: ModuleType | None) -> TypeGuard[Callable[..., Any]]:
    """Check if the given target is from an external python module.

    Args:
        target (TorchTarget): the target of an FX node or a subclass of `torch.nn.Module`
            when the node's op is "call_module".
        module (ModuleType | None): a python module or None when the module is not identified.

    Returns:
        bool: `False` if the module is supported, `True` otherwise. When the return value is `True`,
            it is guaranteed that the target is callable.
    """
    # These checks detect unsupported patterns and do either one of the following.
    # a) Log a precise instruction that users can adhere to and raise an exception.
    # b) Return `True` to notify users with a general unsupported error as patterns are unpredictable and
    #   detailed instructions cannot be given.
    checks: list[Callable[[TorchTarget, ModuleType | None], TypeGuard[Callable[..., Any]]]] = [
        is_foreign_function,
        is_trampoline_autograd_apply,
    ]

    return any(check(target, module) for check in checks)


def is_foreign_function(target: TorchTarget, module: ModuleType | None) -> TypeGuard[Callable[..., Any]]:
    """Check if the given target is from an external python module.

    Args:
        target (TorchTarget): the target of an FX node or a subclass of `torch.nn.Module`
            when the node's op is "call_module".
        module (ModuleType | None): a python module or None when the module is not identified.

    Returns:
        bool: `False` if the module is native (i.e. either from torch or built-in), `True` otherwise.
            When the return value is `True`, it is guaranteed that the target is callable.
    """
    if (
        isinstance(target, str)  # A method of `torch.Tensor` cannot come from an external python module
        or (module is not None and issubmodule(module, (builtins, _operator, torch)))
    ):
        return False

    if inspect.isclass(target) and issubclass(target, torch.nn.Module):
        if module is None:
            log.error(
                "Found an unexpected external subclass of `torch.nn.Module` in the graph "
                f"`{target.__name__}` from an unidentified external module. "
                "Please report this incident to us at https://squeezebits.zendesk.com/hc/en-us/requests/new "
                "including the error message"
            )  # UX
            raise UnsupportedModuleCallError("An external module class from an unidentified module found.")  # UX

        if torch.__version__ < "2.3":
            is_allowed_in_graph = torch._dynamo.allowed_functions.is_allowed(target)
        else:
            # pylint: disable-next=import-outside-toplevel
            from torch._dynamo import trace_rules, variables

            is_allowed_in_graph = (  # pylint: disable-next=no-member
                trace_rules.lookup_callable(target) == variables.TorchInGraphFunctionVariable  # type: ignore
                or trace_rules.lookup(target) == variables.TorchInGraphFunctionVariable
            )

        if is_allowed_in_graph:
            log.error(
                f"Found an external subclass of `torch.nn.Module` in the graph: `{target.__name__}` "
                f"from the external module `{module.__name__}`. "
                "It looks like this class has been explicitly allowed in graph by calling "
                f"`torch._dynamo.allow_in_graph({target.__name__})`. "
                "Please disallow it manually before passing it to one of OwLite APIs as follows:\n"
                "===================================================\n"
                f"import torch._dynamo\n"
                f"from {module.__name__} import {target.__name__}\n"
                f"torch._dynamo.disallow_in_graph({target.__name__})\n"
                "===================================================\n"
            )  # UX
            raise UnsupportedModuleCallError("An external module class allowed in graph found.")  # UX

        log.error(
            "Found an unexpected external subclass of `torch.nn.Module` in the graph "
            f"`{target.__name__}` from the module `{module.__name__}`. "
            "Please report this incident to us at https://squeezebits.zendesk.com/hc/en-us/requests/new "
            "including the error message"
        )  # UX
        raise UnsupportedModuleCallError("Unexpected external module class found.")  # UX

    return True


def is_trampoline_autograd_apply(target: TorchTarget, module: ModuleType | None) -> TypeGuard[Callable[..., Any]]:
    """Check if the given target is a procedure call to custom `torch.autograd` function.

    Args:
        target (TorchTarget): the target of an FX node or a subclass of `torch.nn.Module`
            when the node's op is "call_module".
        module (ModuleType | None): a python module or None when the module is not identified.

    Returns:
        bool: `False` if the target is not a procedure call to custom `torch.autograd` function.
            Always raises an exception otherwise.
    """
    if not callable(target):
        return False

    is_autograd_apply_target: bool
    match torch_version := torch.__version__:
        case _ if "2.1" <= torch_version < "2.3":
            is_autograd_apply_target = (
                module is torch._dynamo.variables.misc and target.__name__ == "trampoline_autograd_apply"
            )
        case _ if "2.3" <= torch_version < "2.5":
            is_autograd_apply_target = target is torch._functorch.autograd_function.autograd_function_apply
        case _:
            raise RuntimeError(f"PyTorch {torch_version} is not supported by OwLite")

    if is_autograd_apply_target:
        log.error(
            "Found a procedure call to custom `torch.autograd` function in the graph. User defined "
            "`torch.autograd` functions are not supported. Please replace them with native PyTorch "
            "implementations."
        )  # UX
        raise UnsupportedAutogradFunctionCallError("Custom autograd function found in graph.")  # UX

    return False


def get_module(target: TorchTarget) -> ModuleType | None:
    """Get the module of the given FX target.

    Args:
        target (TorchTarget): the target of an FX node or a subclass of `torch.nn.Module`
            when the node's op is "call_module".

    Returns:
        ModuleType | None: the python module where the target is defined at. `None` if the python module is not found.
    """
    if target is torch._dynamo.utils.numpy_attr_wrapper or isinstance(
        target,
        (
            torch._dynamo.utils.numpy_method_wrapper
            | torch._dynamo.utils.numpy_operator_wrapper
            | torch._dynamo.utils.numpy_to_tensor_wrapper
        ),
    ):
        return numpy
    try:
        return inspect.getmodule(target)
    # pylint: disable-next=broad-exception-caught
    except Exception as e:
        log.warning(f"Failed to inspect module of the call_function target {targetstr(target)}: {e}")
        return None


def issubmodule(module: ModuleType, module_or_tuple: ModuleType | tuple[ModuleType, ...]) -> bool:
    """Check if the first python module is a submodule of one of the second modules.

    Args:
        module (ModuleType): a python module
        module_or_tuple (ModuleType | tuple[ModuleType, ...]): another python module (or tuple of other python modules)

    Returns:
        bool: `True` if `module` is a submodule of `module_or_tuple`, `False` otherwise.
    """
    if isinstance(module_or_tuple, ModuleType):
        if not module.__name__.startswith(module_or_tuple.__name__):
            return False
        if (
            (submodule_path := getattr(module, "__file__", None)) is not None
            and (module_path := getattr(module_or_tuple, "__file__", None)) is not None
            and os.path.commonprefix([(parent_dir := os.path.dirname(module_path)), submodule_path]) != parent_dir
        ):
            return False
        return True

    return any(issubmodule(module, m) for m in module_or_tuple)

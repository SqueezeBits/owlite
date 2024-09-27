import inspect

from collections import OrderedDict
from enum import IntEnum
from typing import Callable

import torch._dynamo
import torch.fx

from ..core.logger import log
from .config import DISABLE_AUTO_PATCH

class PatchStatus(IntEnum):
    REGISTERED = 0
    APPLIED = 1

class Patch():
    orig_fn: Callable # original function
    patched_fn: Callable # patched version of the function
    orig_fn_path: str # the full qualified name of the original function
    patched_fn_path: str # the full qualified name of the patched function
    status: PatchStatus # status of the patch
    disallow_in_graph: bool # whether to allow the patched function in graph
    hard_patch_target: str | None # patch target to assign the patched function with assign operator(=)

    def __init__(
            self, 
            orig_fn: Callable, 
            patched_fn: Callable,
            disallow_in_graph: bool,
            hard_patch_target: str | None,
        ) -> None:
        self.orig_fn = orig_fn
        self.patched_fn = patched_fn
        self.orig_fn_module = inspect.getmodule(orig_fn)
        self.orig_fn_path = f"{self.orig_fn_module.__name__}.{orig_fn.__name__}"
        self.patched_fn_module = inspect.getmodule(patched_fn)
        self.patched_fn_path = f"{self.patched_fn_module.__name__}.{patched_fn.__name__}"
        self.status = PatchStatus.REGISTERED
        self.hard_patch_target = hard_patch_target
        self.disallow_in_graph = disallow_in_graph

        if self.hard_patch_target:
            log.debug_warning(f"Hard patch for {self.hard_patch_target} detected, "
                            "note that hard patch can result in unexpected outcome")

    def apply(self) -> None:
        if self.status == PatchStatus.APPLIED:
            log.warning("This patch is already applied")
            return

        self.status = PatchStatus.APPLIED

        if self.hard_patch_target:  
            exec(f"{self.hard_patch_target} = self.patched_fn")

        setattr(self.orig_fn_module, self.orig_fn.__name__, self.patched_fn)

        if self.disallow_in_graph:
            torch._dynamo.allow_in_graph(self.patched_fn)
            torch._dynamo.disallow_in_graph(self.patched_fn)

    def rollback(self) -> None:
        if self.status == PatchStatus.REGISTERED:
            log.warning("This patch is not applied yet")
            return
        
        self.status = PatchStatus.REGISTERED

        if self.hard_patch_target:
            exec(f"{self.hard_patch_target} = self.orig_fn")
        
        setattr(self.orig_fn_module, self.orig_fn.__name__, self.orig_fn)
        # TODO: should rollback reset graph allow settings too?


class PatchManager:
    patches: list[Patch] = []

    @classmethod
    def is_registered(cls, fn_or_fn_path: Callable | str):
        return fn_or_fn_path in [
            f for patch in cls.patches for f in [
                patch.orig_fn, patch.patched_fn, patch.orig_fn_path, patch.patched_fn_path
            ]
        ]

    @classmethod
    def register_patch(
        cls, 
        orig_fn: Callable, 
        patched_fn: Callable, 
        hard_patch_target: str | None = None, 
        disallow_in_graph: bool = False,
    ) -> None:
        if cls.is_registered(orig_fn):
            raise Exception(
                f"Patch conflict detected for {orig_fn.__module__}.{orig_fn.__name__}"
            )
        if cls.is_registered(patched_fn):
            raise Exception(
                f"Patch conflict detected for {patched_fn.__module__}.{patched_fn.__name__}"
            )
        
        if patched_fn is not orig_fn:
            patched_fn_module = inspect.getmodule(patched_fn)
            patched_fn_name = patched_fn.__name__
            setattr(inspect.getmodule(orig_fn), orig_fn.__name__, patched_fn)
            cls.patches.append(Patch(orig_fn, patched_fn, disallow_in_graph, hard_patch_target))
            log.debug(
                f"Registered patch: {orig_fn.__module__}.{orig_fn.__name__} -> "
                f"{patched_fn_module.__name__}.{patched_fn_name}"
            )
        else:
            log.warning(
                f"Ignoring vacuous patch for {orig_fn.__module__}.{orig_fn.__name__}"
            )

    @classmethod
    def deregister_patch(cls, fn_or_fn_path: Callable | str) -> None:
        if not cls.is_registered(fn_or_fn_path):
            fn_path = (
                fn_or_fn_path if isinstance(fn_or_fn_path, str) 
                else '.'.join([str(inspect.getmodule(fn_or_fn_path)), fn_or_fn_path.__name__])
            )
            log.warning(f"No patch registered for {fn_path}")
            return
        
        patch_index = [
            fn for patch in cls.patches for fn in [
                patch.orig_fn, patch.patched_fn, patch.orig_fn_path, patch.patched_fn_path
            ]
        ].index(fn_or_fn_path) // 4
        cls.patches.pop(patch_index)

    @classmethod
    def apply_patches(cls) -> None:
        for patch in cls.patches:
            patch.apply()
    
    @classmethod
    def rollback_patches(cls) -> None:
        for patch in cls.patches:
            patch.rollback()


def register_patch(orig_fn: Callable, hard_patch_target: str | None = None, disallow_in_graph: bool = False):
    def wrap(patched_fn: Callable):
        PatchManager.register_patch(orig_fn, patched_fn, hard_patch_target, disallow_in_graph)
        return patched_fn
    
    return wrap


# [SQZB] patch for DataParallel
@register_patch(torch.fx.GraphModule._replicate_for_data_parallel, "torch.fx.GraphModule._replicate_for_data_parallel")
def patched_replicate_for_data_parallel(self):
    new_gm = self.__copy__()
    new_gm._is_replica = True

    # [SQZB] replicas do not have parameters themselves, the replicas reference the original module.
    new_gm._parameters = OrderedDict()
    new_gm._buffers = new_gm._buffers.copy()
    new_gm._modules = new_gm._modules.copy()
    new_gm.graph.owning_module = self

    return new_gm


if not DISABLE_AUTO_PATCH:
    PatchManager.apply_patches()

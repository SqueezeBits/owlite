from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager

from ..config import FX_TRANSFORM_MAXIMUM_ITERATION
from .passes import (
    ConnectInplaceOpsToUsers,
    DecomposeInProjection,
    DecomposeInProjectionPacked,
    DecomposeMultiheadAttention,
    DecomposeMultiHeadAttentionForward,
    DecomposeScaledDotProductAttention,
    DecomposeSiLU,
    DecomposeTransformer,
    DecomposeTransformerDecoder,
    DecomposeTransformerDecoderLayer,
    DecomposeTransformerEncoder,
    DecomposeTransformerEncoderLayer,
    EliminateExplicitGetitem,
    EliminateIdentity,
    FixHardCodedDevice,
    FuseConsecutiveConcats,
    PassName,
)


def optimize(
    graph_module: GraphModule,
    skipped_optimizers: list[PassName] | None = None,
) -> PassResult:
    """Optimize the given graph module inplace.

    Args:
        graph_module (GraphModule): a graph module
        skipped_optimizers (list[PassName] | None, optional): the names of optimization passes to skip.
            Defaults to None.

    Returns:
        PassResult: the result of the transform
    """
    result = get_pass_manager(skipped_optimizers)(graph_module)
    if result.modified:
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        graph_module.recompile()
    return result


def get_pass_manager(skipped_optimizers: list[PassName] | None = None) -> PassManager:
    """Get pass manager.

    Args:
        skipped_optimizers (list[PassName] | None, optional): the names of optimization passes to skip.
            Defaults to None.

    Returns:
        PassManager: a pass manager
    """
    pass_manager = PassManager(steps=FX_TRANSFORM_MAXIMUM_ITERATION)

    functionality_fixes = (
        ConnectInplaceOpsToUsers,
        EliminateIdentity,
        EliminateExplicitGetitem,
        FixHardCodedDevice,
    )
    transformer_rewrite_passes = (
        DecomposeTransformer,
        DecomposeTransformerEncoder,
        DecomposeTransformerEncoderLayer,
        DecomposeTransformerDecoder,
        DecomposeTransformerDecoderLayer,
    )
    mha_rewrite_passes = (
        DecomposeMultiheadAttention,
        DecomposeMultiHeadAttentionForward,
        DecomposeInProjectionPacked,
        DecomposeInProjection,
        DecomposeScaledDotProductAttention,
    )
    other_rewrite_passes = (
        DecomposeSiLU,
        FuseConsecutiveConcats,
    )

    for fx_pass in (
        *functionality_fixes,
        *transformer_rewrite_passes,
        *mha_rewrite_passes,
        *other_rewrite_passes,
    ):
        if skipped_optimizers is not None and fx_pass.__name__ in skipped_optimizers:
            continue
        pass_manager.add_pass(fx_pass())

    return pass_manager

import builtins

import torch.nn.functional as F
from torch.fx.node import Node


def call_canonical_mask(
    mask: Node | None,
    mask_name: str,
    other: Node | None,
    other_name: str,
    target: Node,
    check_other: bool = False,
) -> Node | None:
    """Call `F._canonical_mask` with tensors instead of types.

    In code level, this would be equivalent to
    ```python
    F._canonical_mask(
        mask=mask,
        mask_name=mask_name,
        other_type=other.dtype,
        other_name=other_name,
        target_type=target.dtype,
        check_other=check_other,
    )
    ```

    Args:
        mask (Node | None): a node generating a mask tensor
        mask_name (str): the `mask_name` parameter for the `F._canonical_mask`
        other (Node | None): a reference node for providing `other_type` parameter for the `F._canonical_mask`
        other_name (str): the `other_name` parameter for the `F._canonical_mask`
        target (Node): a reference node for providing `target_type` parameter for the `F._canonical_mask`
        check_other (bool, optional): the `check_other` parameter for the `F._canonical_mask`. Defaults to False.

    Returns:
        Node | None: the output node of the function call `F._canonical_mask`
    """
    graph = target.graph
    if mask is None:
        return None
    other_type: Node | None = None
    if other is not None:
        other_type = graph.call_function(builtins.getattr, (other, "dtype"))
    target_type: Node | None = None
    if target is not None:
        target_type = graph.call_function(builtins.getattr, (target, "dtype"))
    return graph.call_function(
        # pylint: disable-next=protected-access
        F._canonical_mask,
        kwargs={
            "mask": mask,
            "mask_name": mask_name,
            "other_type": other_type,
            "other_name": other_name,
            "target_type": target_type,
            "check_other": check_other,
        },
    )


def inline_get_seq_len(src: Node, batch_first: bool) -> Node:
    """Inline the function call `torch.nn.modules.transformer._get_seq_len`.

    Args:
        src (Node): a node corresponding to the argument `src: Tensor` of `_get_seq_len`
        batch_first (bool): the argument `batch_first: bool` for `_get_seq_len`

    Returns:
        Node: the output node of the inlined function call `_get_seq_len`
    """
    graph = src.graph
    # if not is_batched: src.shape == (S, E)
    # elif batch_first: src.shape == (N, S, E)
    # else: src.shape == (S, N, E)
    return graph.call_method("size", (src, 0 if batch_first else -2))

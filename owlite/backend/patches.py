# pylint: skip-file
# ruff: noqa
# fmt: off
import inspect
import math
import operator

from collections import OrderedDict
from dataclasses import dataclass
from packaging import version
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int

import torch
import torch._dynamo
import torch.nn.functional as F
from torch import Tensor
from torch.fx.node import _side_effectful_functions

from owlite_core.logger import log

try:
    import diffusers
except ImportError:
    diffusers = None

try:
    import torchvision
except ImportError:
    torchvision = None


_side_effectful_functions.add(operator.setitem)


@dataclass
class PatchHistory:
    orig_fn: Callable
    patched_fn: Callable


class PatchManager:
    _history: list[PatchHistory] = []

    @classmethod
    def patch(
        cls,
        orig_fn: Callable,
        patched_fn: Callable,
    ):
        for history in cls._history:
            if (
                orig_fn is history.orig_fn
                or orig_fn is history.patched_fn
                or patched_fn is history.orig_fn
                or patched_fn is history.patched_fn
            ):
                raise Exception(
                    f"Patch conflict detected for {orig_fn.__module__}.{orig_fn.__name__}"
                )
        if patched_fn is not orig_fn:
            patched_fn_module = inspect.getmodule(patched_fn)
            patched_fn_name = patched_fn.__name__
            setattr(inspect.getmodule(orig_fn), orig_fn.__name__, patched_fn)
            cls._history.append(PatchHistory(orig_fn, patched_fn))
            log.debug(
                f"Patched {orig_fn.__module__}.{orig_fn.__name__} by {patched_fn_module.__name__}.{patched_fn_name}"
            )
        else:
            log.warning(
                f"Ignoring vacuous patch for {orig_fn.__module__}{orig_fn.__name__}"
            )

    @classmethod
    def rollback(cls, orig_fn_or_patched_fn: Callable):
        for history in cls._history:
            if (
                history.orig_fn is orig_fn_or_patched_fn
                or history.patched_fn is orig_fn_or_patched_fn
            ):
                setattr(
                    history.patched_fn_module,
                    history.patched_fn_name,
                    history.patched_fn,
                )
                setattr(
                    inspect.getmodule(history.orig_fn),
                    history.orig_fn.__name__,
                    history.orig_fn,
                )
                cls._history.remove(history)
                log.info(
                    f"Rolled back the patched function {history.patched_fn_module.__name__}.{history.patched_fn_name} to {history.orig_fn.__module__}.{history.orig_fn.__name__}"
                )
                return
        log.warning(
            f"No patch registered for {orig_fn_or_patched_fn.__module__}.{orig_fn_or_patched_fn.__name__}"
        )


def force_dynamo_disallow_in_graph(obj):
    """Forcefully do what `torch._dynamo.disallow_in_graph` is supposed to do by
    setting the designated property `torchdynamo_force_dynamic` to `True`.
    See [torch._dynamo.mutation_guard.is_dynamic_nn_module](https://github.com/pytorch/pytorch/blob/7bcf7da3a268b435777fe87c7794c382f444e86d/torch/_dynamo/mutation_guard.py#L84)

    Args:
        obj: an object to disallow in torch dynamo graph
    """
    torch._dynamo.disallow_in_graph(obj)
    obj.torchdynamo_force_dynamic = True


def patch(orig_fn: Callable):
    def wrap(patched_fn: Callable):
        PatchManager.patch(orig_fn, patched_fn)
        return patched_fn

    return wrap


def rollback(orig_fn_or_patched_fn: Callable):
    PatchManager.rollback(orig_fn_or_patched_fn)


# [SQZB] patch for DataParallel
def patched_replicate_for_data_parallel(self):
    new_gm = self.__copy__()
    new_gm._is_replica = True

    # [SQZB] replicas do not have parameters themselves, the replicas reference the original module.
    new_gm._parameters = OrderedDict()
    new_gm._buffers = new_gm._buffers.copy()
    new_gm._modules = new_gm._modules.copy()

    return new_gm


torch.fx.GraphModule._replicate_for_data_parallel = patched_replicate_for_data_parallel

# See https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention
# [SQZB] torch.nn.functional.scaled_dot_product_attention cannot be exported to ONNX
@patch(torch.nn.functional.scaled_dot_product_attention)
def slow_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
) -> Tensor:
    L = query.shape[-2]
    S = key.shape[-2]
    attn_mask = (
        torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        if is_causal
        else attn_mask
    )
    attn_mask = (
        torch.zeros_like(attn_mask, dtype=query.dtype, device=query.device).masked_fill(
            ~attn_mask, float("-inf")
        )
        if attn_mask is not None and attn_mask.dtype == torch.bool
        else attn_mask
    )

    attn_weight = F.softmax(
        (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)))
        + (attn_mask if attn_mask is not None else 0),
        dim=-1,
    )
    attn_weight = F.dropout(attn_weight, dropout_p).to(value.dtype)
    return attn_weight @ value

torch.nn.functional.scaled_dot_product_attention = slow_scaled_dot_product_attention

# [SQZB] torch.nn.functional._mha_shape_check causes the error: "torch.* op returned non-Tensor bool"
# Made it a local function with no changes in its contents from torch==2.1.0 (same as torch==2.0.0)
@patch(torch.nn.functional._mha_shape_check)
def patched_mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
                     key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

    return is_batched


# [SQZB] torch.nn.functional._none_or_dtype causes the error: "torch.* op returned non-Tensor bool"
# Made it a local function with no changes in its contents from torch==2.1.0 (same as torch==2.0.0)
@patch(torch.nn.functional._none_or_dtype)
def patched_none_or_dtype(input: Optional[Tensor]) -> Optional[DType]:
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")


@patch(torch.nn.functional._in_projection_packed)
def patched_in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> list[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if not torch._utils.is_compiling() and (
        k is v
    ):  # [SQZB] we can't tell whether it is self attention or not during torch.compile
        if q is k:
            # self-attention
            proj = F.linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = (
                proj.unflatten(-1, (3, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = F.linear(q, w_q, b_q)
            kv_proj = F.linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = (
                kv_proj.unflatten(-1, (2, E))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


@patch(torch.nn.functional.multi_head_attention_forward)
def patched_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not nedeeded.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal is provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    # [SQZB] has_torch_function causes the error: "torch.* op returned non-Tensor bool"
    # tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    # if has_torch_function(tens_ops):
    #     return handle_torch_function(
    #         multi_head_attention_forward,
    #         tens_ops,
    #         query,
    #         key,
    #         value,
    #         embed_dim_to_check,
    #         num_heads,
    #         in_proj_weight,
    #         in_proj_bias,
    #         bias_k,
    #         bias_v,
    #         add_zero_attn,
    #         dropout_p,
    #         out_proj_weight,
    #         out_proj_bias,
    #         training=training,
    #         key_padding_mask=key_padding_mask,
    #         need_weights=need_weights,
    #         attn_mask=attn_mask,
    #         is_causal=is_causal,
    #         use_separate_proj_weight=use_separate_proj_weight,
    #         q_proj_weight=q_proj_weight,
    #         k_proj_weight=k_proj_weight,
    #         v_proj_weight=v_proj_weight,
    #         static_k=static_k,
    #         static_v=static_v,
    #         average_attn_weights=average_attn_weights,
    #     )

    is_batched = F._mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_heads
    )

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert (
            in_proj_weight is not None
        ), "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert (
            q_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            k_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            v_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = F._in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )

    # prep attention mask

    attn_mask = F._canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=q.dtype,
        check_other=False,
    )

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_k.size(0) == bsz * num_heads
        ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert (
            static_k.size(2) == head_dim
        ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_v.size(0) == bsz * num_heads
        ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert (
            static_v.size(2) == head_dim
        ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)

        assert not (
            is_causal and attn_mask is None
        ), "FIXME: is_causal not implemented for need_weights"

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(
                attn_mask, q_scaled, k.transpose(-2, -1)
            )
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        )
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        # [SQZB] Use slow_scaled_dot_product_attention instead of torch.nn.functional.scaled_dot_product_attention
        # attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = slow_scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )

        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None


torch._dynamo.disallow_in_graph(torch.nn.modules.MultiheadAttention)


def patched_nn_multihead_attention_forward(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    is_batched = query.dim() == 3

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    attn_mask = F._canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=query.dtype,
        check_other=False,
    )

    # [SQZB] disable fast path to avoid unsupported ops, python built-in operator is is not supported by torch dynamo
    # why_not_fast_path = ''
    # if not is_batched:
    #     why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
    # elif query is not key or key is not value:
    #     # When lifting this restriction, don't forget to either
    #     # enforce that the dtypes all match or test cases where
    #     # they don't!
    #     why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
    # elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
    #     why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
    # elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
    #     # this case will fail anyway, but at least they'll get a useful error message.
    #     why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
    # elif self.training:
    #     why_not_fast_path = "training is enabled"
    # elif not self.batch_first:
    #     why_not_fast_path = "batch_first was not True"
    # elif self.bias_k is not None:
    #     why_not_fast_path = "self.bias_k was not None"
    # elif self.bias_v is not None:
    #     why_not_fast_path = "self.bias_v was not None"
    # elif self.add_zero_attn:
    #     why_not_fast_path = "add_zero_attn was enabled"
    # elif not self._qkv_same_embed_dim:
    #     why_not_fast_path = "_qkv_same_embed_dim was not True"
    # elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
    #     why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
    #                          is not supported with NestedTensor input"
    # elif torch.is_autocast_enabled():
    #     why_not_fast_path = "autocast is enabled"

    # if not why_not_fast_path:
    #     tensor_args = (
    #         query,
    #         key,
    #         value,
    #         self.in_proj_weight,
    #         self.in_proj_bias,
    #         self.out_proj.weight,
    #         self.out_proj.bias,
    #     )
    #     # We have to use list comprehensions below because TorchScript does not support
    #     # generator expressions.
    #     if torch.overrides.has_torch_function(tensor_args):
    #         why_not_fast_path = "some Tensor argument has_torch_function"
    #     elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
    #         why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
    #     elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
    #         why_not_fast_path = ("grad is enabled and at least one of query or the "
    #                              "input/output projection weights or biases requires_grad")
    #     if not why_not_fast_path:
    #         merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

    #         return torch._native_multi_head_attention(
    #             query,
    #             key,
    #             value,
    #             self.embed_dim,
    #             self.num_heads,
    #             self.in_proj_weight,
    #             self.in_proj_bias,
    #             self.out_proj.weight,
    #             self.out_proj.bias,
    #             merged_mask,
    #             need_weights,
    #             average_attn_weights,
    #             mask_type)

    # any_nested = query.is_nested or key.is_nested or value.is_nested
    # assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
    #                         f"The fast path was not hit because {why_not_fast_path}")

    if self.batch_first and is_batched:
        # [SQZB] python built-in operator is is not supported by torch dynamo
        if torch._utils.is_compiling():
            query, key, value = [
                x.transpose(1, 0)
                for x in (torch.clone(query), torch.clone(key), torch.clone(value))
            ]
        else:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

    if not self._qkv_same_embed_dim:
        (
            attn_output,
            attn_output_weights,
        ) = patched_multi_head_attention_forward(  # [SQZB] just call patched version directly for clarification
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
    else:
        (
            attn_output,
            attn_output_weights,
        ) = patched_multi_head_attention_forward(  # [SQZB] just call patched version directly for clarification
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
    if self.batch_first and is_batched:
        return attn_output.transpose(1, 0), attn_output_weights
    else:
        return attn_output, attn_output_weights


torch.nn.modules.MultiheadAttention.forward = patched_nn_multihead_attention_forward


if "2.0.0" <= torch.__version__ < "2.1.0":
    # [SQZB] unflatten cannot be exported to ONNX
    @patch(torch.unflatten)
    def patched_unflatten(self, dim, sizes):
        if dim == -1:
            return self.reshape(*self.shape[:-1], *sizes)
        if dim == 0:
            return self.reshape(*sizes, *self.shape[1:])
        return self.reshape(*self.shape[:dim], *sizes, *self.shape[dim + 1 :])


    Tensor.unflatten = patched_unflatten

    torch._dynamo.disallow_in_graph(torch.nn.Transformer)
    torch._dynamo.disallow_in_graph(torch.nn.modules.TransformerEncoder)
    torch._dynamo.disallow_in_graph(torch.nn.modules.TransformerDecoder)
    torch._dynamo.disallow_in_graph(torch.nn.modules.TransformerEncoderLayer)
    torch._dynamo.disallow_in_graph(torch.nn.modules.TransformerDecoderLayer)

    def pathed_nn_transformer_encoder_layer_forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as src_mask.
                Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ""
        if not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or "cpu" in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(
                    src_mask, src_key_padding_mask, src
                )
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )

        x = src
        if self.norm_first:
            # [SQZB] our compiler just can't handle this nesting forward call,
            # it seems to lack an ability to handle a submodule call(self.self_attn) nested inside a method call(self._sa_block)
            # x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            _x = self.norm1(x)
            _x = self.self_attn(
                _x,
                _x,
                _x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0]
            _x = self.dropout1(_x)
            x = x + _x

            # x = x + self._ff_block(self.norm2(x))
            _x = self.norm2(x)
            _x = self.linear2(self.dropout(self.activation(self.linear1(_x))))
            _x = self.dropout2(_x)

            x = x + _x
        else:
            # [SQZB] our compiler just can't handle this nesting forward call,
            # it seems to lack an ability to handle a submodule call(self.self_attn) nested inside a method call(self._sa_block)
            # x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            _x = self.self_attn(
                x,
                x,
                x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0]
            _x = self.dropout1(_x)
            x = self.norm1(x + _x)

            # x = self.norm2(x + self._ff_block(x))
            _x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            _x = self.dropout2(_x)
            x = self.norm2(x + _x)

        return x


    torch.nn.modules.TransformerEncoderLayer.forward = (
        pathed_nn_transformer_encoder_layer_forward
    )


    def patched_nn_transformer_decoder_layer_forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing tgt_mask. Default: ``False``.
            memory_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing memory_mask. Default: ``False``.
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            # [SQZB] our compiler just can't handle this nesting forward call,
            # it seems to lack an ability to handle a submodule call(self.self_attn) nested inside a method call(self._sa_block)
            # x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            _x = self.norm1(x)
            _x = self.self_attn(
                _x,
                _x,
                _x,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=tgt_is_causal,
                need_weights=False,
            )[0]
            _x = self.dropout1(_x)
            x = x + _x

            # x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            _x = self.norm2(x)
            _x = self.multihead_attn(
                _x,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                is_causal=memory_is_causal,
                need_weights=False,
            )[0]
            _x = self.dropout2(_x)
            x = x + _x

            # x = x + self._ff_block(self.norm3(x))
            _x = self.norm3(x)
            _x = self.linear2(self.dropout(self.activation(self.linear1(_x))))
            _x = self.dropout3(_x)
            x = x + _x

        else:
            # [SQZB] our compiler just can't handle this nesting forward call,
            # it seems to lack an ability to handle a submodule call(self.self_attn) nested inside a method call(self._sa_block)
            # x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            _x = self.self_attn(
                x,
                x,
                x,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=tgt_is_causal,
                need_weights=False,
            )[0]
            _x = self.dropout1(_x)
            x = self.norm1(x + _x)

            # x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            _x = self.multihead_attn(
                x,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                is_causal=memory_is_causal,
                need_weights=False,
            )[0]
            _x = self.dropout2(_x)
            x = self.norm2(x + _x)

            # x = self.norm3(x + self._ff_block(x))
            _x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            _x = self.dropout3(_x)
            x = self.norm3(x + _x)

        return x


    torch.nn.modules.TransformerDecoderLayer.forward = (
        patched_nn_transformer_decoder_layer_forward
    )

if "2.1.0" <= torch.__version__ < "2.2.0":
    force_dynamo_disallow_in_graph(torch.nn.Transformer)
    force_dynamo_disallow_in_graph(torch.nn.modules.TransformerEncoder)
    force_dynamo_disallow_in_graph(torch.nn.modules.TransformerDecoder)
    force_dynamo_disallow_in_graph(torch.nn.modules.TransformerEncoderLayer)
    force_dynamo_disallow_in_graph(torch.nn.modules.TransformerDecoderLayer)

    from torch.nn.modules.transformer import _generate_square_subsequent_mask

    # [SQZB] torch.nn.functional._none_or_dtype causes the error: "torch.* op returned non-Tensor bool"
    # Made it a local function with no changes in its contents from torch==2.1.0
    @patch(torch.nn.modules.transformer._detect_is_causal_mask)
    def patched_detect_is_causal_mask(
            mask: Optional[Tensor],
            is_causal: Optional[bool] = None,
            size: Optional[int] = None,
    ) -> bool:
        """Return whether the given attention mask is causal.

        Warning:
        If ``is_causal`` is not ``None``, its value will be returned as is.  If a
        user supplies an incorrect ``is_causal`` hint,

        ``is_causal=False`` when the mask is in fact a causal attention.mask
        may lead to reduced performance relative to what would be achievable
        with ``is_causal=True``;
        ``is_causal=True`` when the mask is in fact not a causal attention.mask
        may lead to incorrect and unpredictable execution - in some scenarios,
        a causal mask may be applied based on the hint, in other execution
        scenarios the specified mask may be used.  The choice may not appear
        to be deterministic, in that a number of factors like alignment,
        hardware SKU, etc influence the decision whether to use a mask or
        rely on the hint.
        ``size`` if not None, check whether the mask is a causal mask of the provided size
        Otherwise, checks for any causal mask.
        """
        # Prevent type refinement
        make_causal = (is_causal is True)

        if is_causal is None and mask is not None:
            sz = size if size is not None else mask.size(-2)
            causal_comparison = _generate_square_subsequent_mask(
                sz, device=mask.device, dtype=mask.dtype)

            # Do not use `torch.equal` so we handle batched masks by
            # broadcasting the comparison.
            if mask.size() == causal_comparison.size():
                make_causal = bool((mask == causal_comparison).all())
            else:
                make_causal = False

        return make_causal


    @patch(torch.nn.modules.transformer._get_seq_len)
    def patched_get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
        # [SQZB] Accessing src.is_nested causes an error when calling the graph module.
        # The error message: "target builtins.getattr has type str but a Callable is expected"
        # if src.is_nested:
        #     return None
        # else:
        #     src_size = src.size()
        #     if len(src_size) == 2:
        #         # unbatched: S, E
        #         return src_size[0]
        #     else:
        #         # batched: B, S, E if batch_first else S, B, E
        #         seq_len_pos = 1 if batch_first else 0
        #         return src_size[seq_len_pos]
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

if (
    torchvision is not None
    and version.parse("0.15.0") <= version.parse(torchvision.__version__) < version.parse("0.16.0")
):
    @patch(torchvision.models.swin_transformer.shifted_window_attention)
    def patched_shifted_window_attention(
        input: Tensor,
        qkv_weight: Tensor,
        proj_weight: Tensor,
        relative_position_bias: Tensor,
        window_size: list[int],
        num_heads: int,
        shift_size: list[int],
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        qkv_bias: Optional[Tensor] = None,
        proj_bias: Optional[Tensor] = None,
        logit_scale: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tensor:
        """
        Window based multi-head self attention (W-MSA) module with relative position bias.
        It supports both of shifted and non-shifted window.
        Args:
            input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
            qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
            proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
            relative_position_bias (Tensor): The learned relative position bias added to attention.
            window_size (List[int]): Window size.
            num_heads (int): Number of attention heads.
            shift_size (List[int]): Shift size for shifted window attention.
            attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
            dropout (float): Dropout ratio of output. Default: 0.0.
            qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
            proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
            logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
            training (bool, optional): Training flag used by the dropout parameters. Default: True.
        Returns:
            Tensor[N, H, W, C]: The output tensor after shifted window attention.
        """
        B, H, W, C = input.shape
        # pad feature maps to multiples of window size
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_H, pad_W, _ = x.shape

        # [SQZB] .copy() method is not supported by torch.compile(fullgraph=True)
        # shift_size = shift_size.copy()
        shift_size = shift_size[:]
        # If window size is larger than feature size, there is no need to shift window
        if window_size[0] >= pad_H:
            shift_size[0] = 0
        if window_size[1] >= pad_W:
            shift_size[1] = 0

        # cyclic shift
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

        # partition windows
        num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
        x = x.view(
            B,
            pad_H // window_size[0],
            window_size[0],
            pad_W // window_size[1],
            window_size[1],
            C,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            B * num_windows, window_size[0] * window_size[1], C
        )  # B*nW, Ws*Ws, C

        # multi-head attention
        if logit_scale is not None and qkv_bias is not None:
            qkv_bias = qkv_bias.clone()
            length = qkv_bias.numel() // 3
            qkv_bias[length : 2 * length].zero_()
        qkv = F.linear(x, qkv_weight, qkv_bias)
        qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if logit_scale is not None:
            # cosine attention
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
            attn = attn * logit_scale
        else:
            q = q * (C // num_heads) ** -0.5
            attn = q.matmul(k.transpose(-2, -1))
        # add relative position bias
        attn = attn + relative_position_bias

        if sum(shift_size) > 0:
            # generate attention mask
            attn_mask = x.new_zeros((pad_H, pad_W))
            h_slices = (
                (0, -window_size[0]),
                (-window_size[0], -shift_size[0]),
                (-shift_size[0], None),
            )
            w_slices = (
                (0, -window_size[1]),
                (-window_size[1], -shift_size[1]),
                (-shift_size[1], None),
            )
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(
                pad_H // window_size[0],
                window_size[0],
                pad_W // window_size[1],
                window_size[1],
            )
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(
                num_windows, window_size[0] * window_size[1]
            )
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, 0.0)
            attn = attn.view(
                x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1)
            )
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, num_heads, x.size(1), x.size(1))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=attention_dropout, training=training)

        x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        x = F.linear(x, proj_weight, proj_bias)
        x = F.dropout(x, p=dropout, training=training)

        # reverse windows
        x = x.view(
            B,
            pad_H // window_size[0],
            pad_W // window_size[1],
            window_size[0],
            window_size[1],
            C,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

        # reverse cyclic shift
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

        # unpad features
        x = x[:, :H, :W, :].contiguous()
        return x

if diffusers is not None:
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    from diffusers.models.attention import BasicTransformerBlock

    @patch(AttnProcessor2_0)
    class PatchedAttnProcessor2_0(torch.nn.Module):
        r"""
        Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
        """

        def __init__(self):
            super().__init__()
            if not hasattr(F, "scaled_dot_product_attention"):
                raise ImportError(
                    "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
                )

        def forward(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
        ):
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)
            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, attn.heads, -1, attention_mask.shape[-1]
                )
            if attn.group_norm is not None:
                hidden_states = attn.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)
            query = attn.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            # [SQZB] Use slow_scaled_dot_product_attention instead of torch.nn.functional.scaled_dot_product_attention
            # hidden_states = F.scaled_dot_product_attention(
            hidden_states = slow_scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states

    torch._dynamo.disallow_in_graph(Attention)
    torch._dynamo.disallow_in_graph(BasicTransformerBlock)

# fmt: on

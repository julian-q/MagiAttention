import math
from collections import namedtuple
from typing import Optional

import torch
from flash_attn import flash_attn_func as fa2_func
from torch.nn.functional import scaled_dot_product_attention as sdpa_func

# -------------------       attn impl utils     ------------------- #


# from flash_attn_interface import _flex_flash_attn_forward, _flex_flash_attn_backward


def sdpa_naive_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: Optional[float] = 0.0,
    is_causal: Optional[bool] = False,
    scale: Optional[float] = None,
    return_attn_probs: Optional[bool] = False,
) -> torch.Tensor:
    """naive pytorch implementation of scaled dot product attention (sdpa)

    Args:
        q: [b, h, sq, d]
        k: [b, h, sk, d]
        v: [b, h, sv, d]

    Returns:
        o: [b, h, sq, d]
    """
    # init
    sq, sk = q.size(-2), k.size(-2)
    scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(sq, sk, dtype=q.dtype).to(q.device)

    # get attn_bias / attn_mask
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(sq, sk, dtype=torch.bool).tril(diagonal=0).to(q.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    # scaled dot for q,k
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    if return_attn_probs:
        lse = torch.logsumexp(attn_weight, dim=-1)

    # softmax
    attn_weight = torch.softmax(attn_weight, dim=-1)

    # dropout
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    # weighted sum with v
    attn_out = attn_weight @ v

    if return_attn_probs:
        return attn_out, lse.to(q.dtype)
    return attn_out


attn_impls = namedtuple(
    "AttnImpls",
    field_names=[
        "fa2_func",
        "sdpa_func",
        "sdpa_naive_func",
    ],
    defaults=[
        fa2_func,
        sdpa_func,
        sdpa_naive_func,
    ],
)()

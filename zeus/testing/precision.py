import re

import torch
from einops import rearrange


def assert_close(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    mismatch_threshold: float = 0,
) -> None:
    assert 0 <= mismatch_threshold <= 1, "mismatch_threshold must be between 0 and 1"
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    except AssertionError as e:
        error_msg = str(e)
        if mismatch_threshold > 0:
            # 使用正则表达式提取百分比
            match = re.search(
                r"Mismatched elements: \d+ / \d+ \(([\d.]+)%\)", error_msg
            )
            if match:
                mismatch_ratio = float(match.group(1)) / 100  # 将百分比转换为小数
                print(f"mismatch_ratio: {mismatch_ratio}")
                if mismatch_ratio <= mismatch_threshold:
                    return  # 如果在允许范围内，则不抛出异常
        raise e


def get_mask_from_ranges(
    q_ranges: list[tuple[int, int]],
    k_ranges: list[tuple[int, int]],
    q_len: int,
    k_len: int,
    is_causal_mapping: list[bool],
):
    assert all((is_causal is False) for is_causal in is_causal_mapping)

    bsz = len(q_ranges)
    mask = torch.zeros(
        (q_len, k_len), device=torch.cuda.current_device(), dtype=torch.bool
    )
    for i in range(bsz):
        mask[q_ranges[i][0] : q_ranges[i][1], k_ranges[i][0] : k_ranges[i][1]] = True
    return mask


def torch_attn_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    layout: str = "thd",
    high_precision: bool = False,
) -> torch.Tensor:
    if layout == "thd":
        q = rearrange(q, "t h d -> 1 h t d")
        k = rearrange(k, "t h d -> 1 h t d")
        v = rearrange(v, "t h d -> 1 h t d")
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if high_precision:
        out = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), attn_mask=mask
        )
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    if layout == "thd":
        out = rearrange(out, "1 h t d -> t h d")
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if high_precision:
        return out.to(q.dtype)
    else:
        return out

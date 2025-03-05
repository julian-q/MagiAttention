import re

import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch.nn.attention import SDPBackend, sdpa_kernel

from zeus.common.ranges import NaiveRanges

if version.parse(torch.__version__) > version.parse("2.4"):
    # NOTE: in testing, we should explicitly allow bf16/fp16 reduction for sdpa
    # by setting `torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)`
    # due to the new feature since torch2.5:
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-in-scaled-dot-product-attention-sdpa
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

# usage: to avoid division by zero in numerical calculation and assert-close testing
EPSILON = 1e-8


def extract_mismatch_info(error_msg: str) -> tuple[int, int, float]:
    match = re.search(r"Mismatched elements: (\d+) / (\d+)", error_msg)

    if match:
        mismatched_elements = int(match.group(1))
        total_elements = int(match.group(2))
        mismatch_ratio = mismatched_elements / total_elements
        return mismatched_elements, total_elements, mismatch_ratio
    else:
        raise ValueError(f"Could not find mismatch elements in {error_msg=}")


def assert_close(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    mismatch_threshold: float = 0,
    test_case: str = "",
) -> None:
    assert (
        0 <= mismatch_threshold <= 1
    ), f"{mismatch_threshold=} must be between 0 and 1"
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    except AssertionError as e:
        error_msg = str(e)
        if mismatch_threshold > 0:
            mismatched_elements, total_elements, mismatch_ratio = extract_mismatch_info(
                error_msg
            )

            mismatch_info = (
                f"[{test_case}]: mismatch_ratio = {mismatched_elements} / {total_elements} "
                f"= {mismatch_ratio * 100:.4f} % | mismatch_threshold={mismatch_threshold * 100:.2f} %"
            )

            if mismatch_ratio <= mismatch_threshold:
                print(mismatch_info)
                return
        raise type(e)(
            f"\n>>>>>>>  Torch Error Message: \n\n{error_msg}\n\n"
            f">>>>>>>  Mismatch Detailed Info: \n\n{mismatch_info}\n\n"
        ) from e


def get_mask_from_ranges(
    q_ranges: NaiveRanges,
    k_ranges: NaiveRanges,
    q_len: int,
    k_len: int,
    is_causal_mapping: list[bool],
) -> torch.Tensor:
    assert all((is_causal is False) for is_causal in is_causal_mapping)

    bsz = len(q_ranges)
    mask = torch.zeros(
        (q_len, k_len), device=torch.cuda.current_device(), dtype=torch.bool
    )
    for i in range(bsz):
        mask[q_ranges[i][0] : q_ranges[i][1], k_ranges[i][0] : k_ranges[i][1]] = True
    return mask


def calc_inf_norm(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    return (a.float() - b.float()).norm(p=float("inf")).item()


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

    with sdpa_kernel(backends=[SDPBackend.MATH]):
        if high_precision:
            out = F.scaled_dot_product_attention(
                q.to(torch.float64),  # NOTE: use fp64 as ground-truth
                k.to(torch.float64),
                v.to(torch.float64),
                attn_mask=mask,
            )
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    if layout == "thd":
        out = rearrange(out, "1 h t d -> t h d")
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if high_precision:
        return out.to(q.dtype)
    else:
        return out

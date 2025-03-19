import torch


def safe_subtract(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Safely subtracts two tensors,
    where the subtraction results of two -inf will be set to -inf.
    """

    eq = (a == b) & (a == float("-inf"))
    sub = a - b
    sub = torch.where(eq, torch.fill(sub, float("-inf")), sub)

    return sub


def softmax_bwd(dout: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Standard backward func for `out = softmax(inp)`"""

    diag_out = torch.diag_embed(out)
    outer_out = torch.einsum("...ij, ...ik -> ...ijk", out, out)

    dinp = torch.einsum("...ij, ...ijk -> ...ik", dout, diag_out - outer_out)

    return dinp

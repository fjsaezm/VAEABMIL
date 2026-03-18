import torch
import numpy as np


class LazyLinear(torch.nn.Module):
    """
    Lazy Linear layer. Extends `torch.nn.Linear` with lazy initialization.
    """

    def __init__(
        self, in_features=None, out_features=512, bias=True, device=None, dtype=None
    ):
        super().__init__()

        if in_features is not None:
            self.module = torch.nn.Linear(
                in_features, out_features, bias=bias, device=device, dtype=dtype
            )
        else:
            self.module = torch.nn.LazyLinear(
                out_features, bias=bias, device=device, dtype=dtype
            )

    def forward(self, x):
        return self.module(x)


def masked_softmax(
    X: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute masked softmax along the second dimension.

    Arguments:
        X (Tensor): Input tensor of shape `(batch_size, N, ...)`.
        mask (Tensor): Mask of shape `(batch_size, N)`. If None, no masking is applied.

    Returns:
        Tensor: Masked softmax of shape `(batch_size, N, ...)`.
    """

    if mask is None:
        return torch.nn.functional.softmax(X, dim=1)

    # Ensure mask is of the same shape as X
    if mask.dim() < X.dim():
        mask = mask.unsqueeze(-1)

    # exp_X = torch.exp(X)
    # exp_X_masked = exp_X * mask
    # sum_exp_X_masked = exp_X_masked.sum(dim=1, keepdim=True)
    # softmax_X = exp_X_masked / (sum_exp_X_masked + 1e-8)
    # return softmax_X

    X_masked = X.masked_fill(mask == 0, -float("inf"))

    return torch.nn.functional.softmax(X_masked, dim=1)


def get_feat_dim(feat_ext: torch.nn.Module, input_shape: tuple[int, ...]) -> int:
    """
    Get feature dimension of a feature extractor.

    Arguments:
        feat_ext (torch.nn.Module): Feature extractor.
        input_shape (tuple): Input shape of the feature extractor.
    """
    with torch.no_grad():
        return feat_ext(torch.zeros((1, *input_shape))).shape[-1]
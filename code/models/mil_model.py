from typing import Any

import torch


def get_args_names(fn):
    args_names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    # remove self from arg_names if exists
    if "self" in args_names:
        args_names = args_names[1:]
    return args_names


class MILModel(torch.nn.Module):
    r"""
    Base class for Multiple Instance Learning (MIL) models in torchmil.

    Subclasses should implement the following methods:

    - `forward`: Forward pass of the model. Accepts bag features (and optionally other arguments) and returns the bag label prediction (and optionally other outputs).
    - `compute_loss`: Compute inner losses of the model. Accepts bag features (and optionally other arguments) and returns the output of the forward method a dictionary of pairs (loss_name, loss_value). By default, the model has no inner losses, so this dictionary is empty.
    - `predict`: Predict bag and (optionally) instance labels. Accepts bag features (and optionally other arguments) and returns label predictions (and optionally instance label predictions).
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the module.
        """
        super(MILModel, self).__init__()

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.

        Returns:
            Y_pred: Bag label prediction of shape `(batch_size,)`.
        """
        raise NotImplementedError

    def compute_loss(
        self, Y: torch.Tensor, X: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, dict]:
        """
        Arguments:
            Y: Bag labels of shape `(batch_size,)`.
            X: Bag features of shape `(batch_size, bag_size, ...)`.

        Returns:
            Y_pred: Bag label prediction of shape `(batch_size,)`.
            loss_dict: Dictionary containing the loss values.
        """

        out = self.forward(X, *args, **kwargs)
        return out, {}

    def predict(
        self, X: torch.Tensor, return_inst_pred: bool = False, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            X: Bag features of shape `(batch_size, bag_size, ...)`.

        Returns:
            Y_pred: Bag label prediction of shape `(batch_size,)`.
            y_inst_pred: If `return_inst_pred=True`, returns instance labels predictions of shape `(batch_size, bag_size)`.
        """
        raise NotImplementedError
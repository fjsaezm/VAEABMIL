import torch
from torch import Tensor

from .utils import masked_softmax, LazyLinear


class AttentionPool(torch.nn.Module):
    r"""
    Attention-based pooling, as proposed in the paper [Attention-based Multiple Instance Learning](https://arxiv.org/abs/1802.04712).

    Given an input bag $\mathbf{X} = \left[ \mathbf{x}_1, \ldots, \mathbf{x}_N \right]^\top \in \mathbb{R}^{N \times \texttt{in_dim}}$,
    this model aggregates the instance features into a bag representation $\mathbf{z} \in \mathbb{R}^{\texttt{in_dim}}$ as,

    $$ \mathbf{z} = \mathbf{X}^\top \operatorname{Softmax}(\mathbf{f}) = \sum_{n=1}^N s_n \mathbf{x}_n, $$

    where $\mathbf{f} = \operatorname{MLP}(\mathbf{X}) \in \mathbb{R}^{N}$ are the attention values and $s_n$ is the normalized attention score for the $n$-th instance.

    To compute the attention values, the $\operatorname{MLP}$ is defined as

    \begin{equation}
    \operatorname{MLP}(\mathbf{X}) = \begin{cases}
    \operatorname{act}(\mathbf{X}\mathbf{W}_1)\mathbf{w}, & \text{if }\texttt{gated=False}, \\
    \left(\operatorname{act}(\mathbf{X}\mathbf{W}_1)\odot\operatorname{sigm}(\mathbf{X}\mathbf{W}_2)\right)\mathbf{w}, & \text{if }\texttt{gated=True},
    \end{cases}
    \end{equation}

    where $\mathbf{W}_1 \in \mathbb{R}^{\texttt{in_dim} \times \texttt{att_dim}}$, $\mathbf{W}_2 \in \mathbb{R}^{\texttt{in_dim} \times \texttt{att_dim}}$,
    $\mathbf{w} \in \mathbb{R}^{\texttt{att_dim}}$, $\operatorname{act} \ \colon \mathbb{R} \to \mathbb{R}$ is the activation function,
    $\operatorname{sigm} \ \colon \mathbb{R} \to \left] 0, 1 \right[$ is the sigmoid function, and $\odot$ denotes element-wise multiplication.
    """

    def __init__(
        self,
        in_dim: int = None,
        att_dim: int = 128,
        act: str = "tanh",
        gated: bool = False,
    ) -> None:
        """
        Arguments:
            in_dim: Input dimension. If not provided, it will be lazily initialized.
            att_dim: Attention dimension.
            act: Activation function for attention. Possible values: 'tanh', 'relu', 'gelu'.
            gated: If True, use gated attention.
        """

        super(AttentionPool, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.act = act
        self.gated = gated

        self.fc1 = LazyLinear(in_dim, att_dim)
        self.fc2 = torch.nn.Linear(att_dim, 1, bias=False)

        if self.gated:
            self.fc_gated = LazyLinear(in_dim, att_dim)
            self.act_gated = torch.nn.Sigmoid()

        if self.act == "tanh":
            self.act_layer = torch.nn.Tanh()
        elif self.act == "relu":
            self.act_layer = torch.nn.ReLU()
        elif self.act == "gelu":
            self.act_layer = torch.nn.GELU()
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] act must be 'tanh', 'relu' or 'gelu'"
            )

    def forward(
        self, X: Tensor, mask: Tensor = None, return_att: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Arguments:
            X: Bag features of shape `(batch_size, bag_size, in_dim)`.
            mask: Mask of shape `(batch_size, bag_size)`.
            return_att: If True, returns attention values (before normalization) in addition to `z`.

        Returns:
            z: Bag representation of shape `(batch_size, in_dim)`.
            f: Only returned when `return_att=True`. Attention values (before normalization) of shape (batch_size, bag_size).
        """

        batch_size = X.shape[0]
        bag_size = X.shape[1]

        if mask is None:
            mask = torch.ones(batch_size, bag_size, device=X.device)
        mask = mask.unsqueeze(dim=-1)  # (batch_size, bag_size, 1)

        H = self.fc1(X)  # (batch_size, bag_size, att_dim)
        H = self.act_layer(H)  # (batch_size, bag_size, att_dim)

        if self.gated:
            G = self.fc_gated(X)
            G = self.act_gated(G)
            H = H * G

        f = self.fc2(H)  # (batch_size, bag_size, 1)

        s = masked_softmax(f, mask)  # (batch_size, bag_size, 1)
        z = torch.bmm(X.transpose(1, 2), s).squeeze(dim=-1)  # (batch_size, in_dim)

        if return_att:
            return z, f.squeeze(dim=-1)
        else:
            return z
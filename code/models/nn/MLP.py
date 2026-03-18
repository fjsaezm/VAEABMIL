import torch


def get_activation(name):
    if "relu" in name:
        return torch.nn.ReLU()
    elif "sigmoid" in name:
        return torch.nn.Sigmoid()
    elif "tanh" in name:
        return torch.nn.Tanh()
    else:
        return None


class MLP(torch.nn.Module):
    def __init__(
        self, input_size=512, linear_sizes=[100, 50], activations=["relu", "relu"]
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.linear_sizes = linear_sizes
        self.activations = activations
        layers = [
            torch.nn.Linear(self.input_size, linear_sizes[0]),
            get_activation(activations[0]),
        ]
        for i in range(1, len(linear_sizes)):
            layers.append(torch.nn.Linear(linear_sizes[i - 1], linear_sizes[i]))
            if activations[i] not in ["None", "none"]:
                layers.append(get_activation(activations[i]))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, X):
        """Computes the output of a multiple linear layer model

        Args:
            X (torch.Tensor): (B, D)

        Returns:
            _type_: _description_
        """
        return self.net(X)


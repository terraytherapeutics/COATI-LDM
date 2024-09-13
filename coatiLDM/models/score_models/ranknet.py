# models and such. Basic MLP. use DUE later.
from typing import Tuple

import torch
from torch import nn


class RankNet(nn.Module):
    def __init__(
        self,
        input_size: int = 2048,
        hidden_size: int = 256,
        n_layers: int = 3,
        dropout_p: float = 0.0,
    ) -> None:
        """Basic RankNet implementation. Pairs of samples are classified
        according to sigmoid(s_i - s_j) where s_i, s_j are scores learned
        during training.

        Args:
            input_size (int, optional): Descriptor size for each sample. Defaults to 2048.
            hidden_size (int, optional): Number of neurons in hidden layers. Defaults to 256.
            n_layers (int, optional): Number of hidden layers. Defaults to 3.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        """
        super(RankNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Dropout(dropout_p), nn.ReLU()
        )

        for _ in range(n_layers):
            self.encoder.append(nn.Linear(hidden_size, hidden_size))
            self.encoder.append(nn.Dropout(dropout_p))
            self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(hidden_size, 1))

    def forward(
        self, x_i: torch.Tensor, x_j: torch.Tensor, sigmoid: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x_i.size() == x_j.size()

        score_i, score_j = self.encoder(x_i), self.encoder(x_j)
        out = score_i - score_j
        if sigmoid:
            out = torch.sigmoid(out)
        return score_i, score_j, out

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Scores sample `x`

        Args:
            x: input fingerprints // (n_samples, n_feat)
        """
        with torch.inference_mode():
            return self.encoder(x)

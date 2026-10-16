import torch
from torch import nn


class RousePINN(nn.Module):
    def __init__(self, n, hidden=64):
        super().__init__()

        self.n = n

        # try a CNN or GNN
        self.net = nn.Sequential(nn.Linear(n * n, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n * 3))

    def forward(self, hic_matrices):
        r = torch.reshape(self.net(torch.flatten(hic_matrices, start_dim=1)), (-1, self.n, 3))

        return r  # + torch.normal(mean=0, std=1, size=r.shape)

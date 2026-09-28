"""MIONet operator network (single-trunk, Cartesian-product form).

Clean re-implementation of the Manifold-Function-Encoder MIONet.  The network
has one branch FNN per encoded input function plus one trunk FNN over the query
coordinates; the output at a point is

    u(x) ~ sum_j  ( prod_b  branch_b(z_b)_j ) * trunk(x)_j   (+ bias).

FNN size convention (matching the original):
    [n0, n1, ...]  linear layers n_{i-1} -> n_i with ReLU between (none after last);
    a NEGATIVE size  -> that layer has no bias;
    a trailing 0     -> identity output (no final linear layer).
"""

from __future__ import annotations

import torch
import torch.nn as nn

ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class FNN(nn.Module):
    def __init__(self, size: list[int], activation: str = "relu"):
        super().__init__()
        self.size = size
        self.act = ACTS[activation]()
        layers = nn.ModuleList()
        for i in range(1, len(size)):
            if size[i] != 0:
                layers.append(nn.Linear(abs(size[i - 1]), abs(size[i]), bias=size[i] > 0))
            else:
                layers.append(None)
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.size) - 2):
            x = self.act(self.layers[i](x))
        last = self.layers[-1]
        return x if last is None else last(x)


class MIONet(nn.Module):
    """Inputs: (branch_1, ..., branch_{B}, positions).  Output: u at positions.

    branch_b : [batch, sensors_b]     positions : [batch, num_loc, dim]
    output   : [batch, num_loc]
    """

    def __init__(self, sizes: list[list[int]], activation: str = "relu", bias: bool = False):
        super().__init__()
        self.nets = nn.ModuleList([FNN(s, activation) for s in sizes])
        self.use_bias = bias
        self.bias = nn.Parameter(torch.zeros(1)) if bias else None

    def forward(self, inputs) -> torch.Tensor:
        branches = torch.stack([self.nets[i](inputs[i]) for i in range(len(self.nets) - 1)])  # [B, batch, p]
        y1 = torch.prod(branches, dim=0)  # [batch, p]
        pts = inputs[-1]
        y2 = self.nets[-1](pts)  # [batch, num_loc, p] or [num_loc, p]
        if y2.dim() == 3:
            y = torch.einsum("ij,ikj->ik", y1, y2)
        else:
            y = y1 @ y2.t()
        return y + self.bias if self.use_bias else y

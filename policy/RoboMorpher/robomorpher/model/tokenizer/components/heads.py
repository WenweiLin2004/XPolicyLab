"""Reconstruction head for the joint residual-VQ decoder."""
import torch.nn as nn
from .embeddings import ResMlpBlock

class DecodeHead(nn.Module):
    def __init__(self, width: int, hidden: int, out_dim: int):
        super().__init__()
        self.block = ResMlpBlock(width, hidden)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.head(self.block(x))

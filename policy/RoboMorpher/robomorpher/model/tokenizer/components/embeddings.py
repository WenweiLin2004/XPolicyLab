"""Per-joint numeric embeddings shared by tokenizer architectures."""
import torch.nn as nn

class ResMlpBlock(nn.Module):
    """x + W2 GELU(W1 LN(x))."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class JointEmbed(nn.Module):
    """Per-joint Linear + residual MLP. Slot identity is left to URDF features
    + causal mask + spatial graph bias — no absolute joint-index embedding."""

    def __init__(self, in_dim: int, width: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, width)
        self.act = nn.GELU(approximate="tanh")
        self.block = ResMlpBlock(width, hidden)

    def forward(self, x):
        return self.block(self.act(self.proj(x)))


class AngleEmbed(nn.Module):
    """Lift scalar angle to width. Default Linear init (not zero)."""

    def __init__(self, width: int, hidden: int = 256, in_dim: int = 1):
        super().__init__()
        hidden = int(hidden)
        if hidden <= 0:
            self.net = nn.Linear(in_dim, width)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden, width),
            )

    def forward(self, angle):
        return self.net(angle)

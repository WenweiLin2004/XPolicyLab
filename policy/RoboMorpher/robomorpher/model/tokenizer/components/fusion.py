"""Direct, independently normalized geometry and angle fusion."""
from torch import nn
from .attention import LlamaRMSNorm


class NormalizedAddFusion(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.urdf_norm = LlamaRMSNorm(width)
        self.angle_norm = LlamaRMSNorm(width)

    def forward(self, urdf, angle, valid=None):
        if urdf.shape != angle.shape:
            raise ValueError("Fusion streams must have identical [B,J,H] shapes")
        # No sqrt(2): the existing state block immediately applies RMSNorm.
        merged = self.urdf_norm(urdf) + self.angle_norm(angle)
        if valid is not None:
            merged = merged.masked_fill(~valid[..., None], 0.0)
        return merged

"""Attention blocks: RMSNorm, SwiGLU MLP, graph-biased MHA, ProcessLayer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# FlexAttention was added after the older 2.4.x torch build used by some
# RoboTwin hosts. Keep it optional: SDPA/eager attention is the correctness
# path and works on Ampere (A800), Ada (4090) and Blackwell (5090).
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    _HAS_FLEX_ATTENTION = True
except (ImportError, ModuleNotFoundError):
    create_block_mask = None
    flex_attention = None
    _HAS_FLEX_ATTENTION = False
from torch.nn.functional import scaled_dot_product_attention
from transformers.integrations import use_kernel_forward_from_hub


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, out_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, out_size, bias=False)
        self.down_proj = nn.Linear(out_size, out_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ChainGraphBias(nn.Module):
    """Graphormer-style attention bias on a serial chain (joint 0 = base).

    Each term is a per-head embedding over graph distance:
      spatial |j_q - j_k|, parent (j_q - j_k)>0, child (j_k - j_q)>0.
    """

    def __init__(
        self,
        num_joints: int,
        num_heads: int,
        enable_spatial: bool = True,
        enable_parent: bool = True,
        enable_child: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.num_heads = num_heads
        self.enable_spatial = enable_spatial
        self.enable_parent = enable_parent
        self.enable_child = enable_child

        joint = torch.arange(num_joints)
        max_dist = num_joints - 1
        self.register_buffer("spd_matrix", (joint[None] - joint[:, None]).abs().long(), persistent=False)
        self.register_buffer("parent_matrix", (joint[:, None] - joint[None]).clamp(min=0).long(), persistent=False)
        self.register_buffer("child_matrix", (joint[None] - joint[:, None]).clamp(min=0).long(), persistent=False)

        def _emb(vocab: int) -> nn.Embedding:
            emb = nn.Embedding(vocab, num_heads, padding_idx=0)
            nn.init.normal_(emb.weight, std=0.02)
            emb.weight.data[0].zero_()
            if zero_init:
                emb.weight.data.zero_()
            return emb

        if enable_spatial:
            self.spatial_embedding = _emb(max_dist + 2)
        if enable_parent:
            self.parent_embedding = _emb(max_dist + 1)
        if enable_child:
            self.child_embedding = _emb(max_dist + 1)

    def joint_bias(self) -> torch.Tensor:
        """(n_head, J, J)."""
        J, H, dev = self.num_joints, self.num_heads, self.spd_matrix.device
        bias = torch.zeros(J, J, H, device=dev)
        if self.enable_spatial:
            bias = bias + self.spatial_embedding(self.spd_matrix + 1)
        if self.enable_parent:
            bias = bias + self.parent_embedding(self.parent_matrix)
        if self.enable_child:
            bias = bias + self.child_embedding(self.child_matrix)
        return bias.permute(2, 0, 1).contiguous()

    def forward(self, L: int) -> torch.Tensor:
        """(n_head, L, L); L must be a multiple of J."""
        J = self.num_joints
        assert L % J == 0, f"sequence length {L} must be a multiple of num_joints {J}"
        ji = torch.arange(L, device=self.spd_matrix.device) % J
        bias_J = self.joint_bias()
        return bias_J.index_select(1, ji).index_select(2, ji)


class LlamaAttention(nn.Module):
    """MHA with optional additive graph bias and bool key mask."""

    def __init__(self, hidden_size, out_size, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = int(out_size / num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, out_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, out_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, out_size, bias=False)
        self.o_proj = nn.Linear(out_size, out_size, bias=False)

    def _eager_attention(self, q, k, v, attn_mask, graph_bias):
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        if graph_bias is not None:
            scores = scores + graph_bias.unsqueeze(0)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                while attn_mask.dim() < scores.dim():
                    attn_mask = attn_mask.unsqueeze(-3 if attn_mask.dim() == 3 else 0)
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(attn, v)

    def _flex_attention(self, q, k, v, attn_mask, graph_bias):
        B, _, L, _ = q.shape
        if graph_bias is not None:
            bias = graph_bias

            def score_mod(score, b, h, q_idx, k_idx):
                return score + bias[h, q_idx, k_idx]
        else:
            score_mod = None

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            mask = attn_mask

            def mask_mod(b, h, q_idx, k_idx):
                # Keep every index a Tensor: a Python 0 mixed with vmap's
                # batched q/k indices fails for singleton/broadcast masks.
                mask_batch = b % mask.shape[0]
                return ~mask[mask_batch, q_idx, k_idx]

            block_mask = create_block_mask(mask_mod, B=B, H=None, Q_LEN=L, KV_LEN=L, device=q.device)
        else:
            block_mask = None
        return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask, scale=self.scaling)

    def forward(self, hidden_states, attn_mask=None, graph_bias=None, use_flex=True):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if graph_bias is None:
            # Internal bool masks use True=blocked; SDPA uses True=allowed.
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                attn_mask = ~attn_mask
            if attn_mask is not None and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_output = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        elif use_flex and _HAS_FLEX_ATTENTION:
            attn_output = self._flex_attention(q, k, v, attn_mask, graph_bias)
        else:
            # Torch < 2.5 has no flex_attention module. The eager path preserves
            # graph bias and masking semantics without requiring a version-specific
            # compiled kernel.
            attn_output = self._eager_attention(q, k, v, attn_mask, graph_bias)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None


class ProcessLayer(nn.Module):
    """One attention + two SwiGLU MLPs. Each layer has its own graph bias."""

    def __init__(
        self,
        hidden_size,
        out_size,
        num_attention_heads,
        use_zero_init_gate: bool = False,
        graph_bias: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.use_zero_init_gate = use_zero_init_gate
        self.graph_bias = graph_bias
        self.attn = LlamaAttention(hidden_size, out_size, num_attention_heads)
        self.mlp_input = LlamaMLP(hidden_size, out_size)
        self.mlp_hidden = LlamaMLP(out_size, out_size)
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(out_size)
        if use_zero_init_gate:
            self.attn_gate = nn.Parameter(torch.zeros(1))
            self.mlp_gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, attn_mask=None):
        bias = None if self.graph_bias is None else self.graph_bias(hidden_states.shape[1])
        hidden_states = self.input_layernorm(hidden_states)
        residual = hidden_states
        hidden_states, _ = self.attn(hidden_states, attn_mask=attn_mask, graph_bias=bias, use_flex=True)
        if self.use_zero_init_gate:
            hidden_states = self.attn_gate * hidden_states
        hidden_states = self.mlp_input(residual) + hidden_states

        residual = hidden_states
        hidden_states = self.mlp_hidden(self.post_attention_layernorm(hidden_states))
        if self.use_zero_init_gate:
            hidden_states = self.mlp_gate * hidden_states
        return residual + hidden_states


class StandardProcessLayer(nn.Module):
    """Standard pre-norm Transformer block with a direct identity residual.

    ``ProcessLayer`` is the MLP-attention residual selected by
    ``block_style: mlp_attention_residual``.
    """

    def __init__(
        self,
        hidden_size,
        out_size,
        num_attention_heads,
        use_zero_init_gate: bool = False,
        graph_bias: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.use_zero_init_gate = use_zero_init_gate
        self.graph_bias = graph_bias
        self.attn = LlamaAttention(hidden_size, out_size, num_attention_heads)
        self.residual_proj = (
            nn.Identity() if hidden_size == out_size
            else nn.Linear(hidden_size, out_size, bias=False)
        )
        self.mlp = LlamaMLP(out_size, out_size)
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(out_size)
        if use_zero_init_gate:
            self.attn_gate = nn.Parameter(torch.zeros(1))
            self.mlp_gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, attn_mask=None):
        bias = None if self.graph_bias is None else self.graph_bias(hidden_states.shape[1])
        residual = self.residual_proj(hidden_states)
        attn_output, _ = self.attn(
            self.input_layernorm(hidden_states),
            attn_mask=attn_mask,
            graph_bias=bias,
            use_flex=True,
        )
        if self.use_zero_init_gate:
            attn_output = self.attn_gate * attn_output
        hidden_states = residual + attn_output

        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        if self.use_zero_init_gate:
            mlp_output = self.mlp_gate * mlp_output
        return hidden_states + mlp_output



class AdaLNZeroProcessLayer(nn.Module):
    """Joint-aligned DiT-style conditioning with one zero-initialized gate per branch.

    Conditioning modulates normalized node features; the existing graph bias is
    independently added to scaled attention logits. Bias zero initialization is
    compatible with AdaLN-Zero and is not a second multiplicative branch gate.
    """

    def __init__(self, hidden_size, out_size, num_attention_heads,
                 use_zero_init_gate=False, graph_bias=None, condition_dim=None):
        super().__init__()
        if hidden_size != out_size:
            raise ValueError("AdaLN-Zero requires hidden_size == out_size")
        if hidden_size < 1 or num_attention_heads < 1 or hidden_size % num_attention_heads:
            raise ValueError("AdaLN-Zero width must be positive and divisible by attention heads")
        if condition_dim is None or int(condition_dim) < 1:
            raise ValueError("AdaLN-Zero requires a positive condition_dim")
        if use_zero_init_gate:
            raise ValueError("AdaLN-Zero cannot be combined with zero_init_gate: two zero gates would block learning")
        self.hidden_size = int(hidden_size)
        self.condition_dim = int(condition_dim)
        self.use_zero_init_gate = False
        self.graph_bias = graph_bias
        self.attn = LlamaAttention(hidden_size, out_size, num_attention_heads)
        self.mlp = LlamaMLP(out_size, out_size)
        self.input_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.post_attention_layernorm = nn.LayerNorm(out_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.condition_dim, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _attention_mask(self, attn_mask, valid):
        """Merge invalid keys with the caller's causal/bidirectional mask."""
        batch, length = valid.shape
        blocked = (~valid)[:, None, :].expand(-1, length, -1)
        if attn_mask is None:
            return blocked
        if not torch.is_tensor(attn_mask) or attn_mask.device != valid.device:
            raise ValueError("Attention mask must be a tensor on the hidden-state device")
        if attn_mask.dtype != torch.bool and not torch.is_floating_point(attn_mask):
            raise ValueError("Attention mask must be boolean or floating point")
        if attn_mask.ndim == 2 and tuple(attn_mask.shape) == (length, length):
            attn_mask = attn_mask.unsqueeze(0)
        if attn_mask.ndim == 3:
            if attn_mask.shape[0] not in (1, batch) or tuple(attn_mask.shape[-2:]) != (length, length):
                raise ValueError("Attention mask must broadcast to [B,J,J]")
            attn_mask = attn_mask.expand(batch, -1, -1)
        elif attn_mask.ndim == 4:
            if (attn_mask.shape[0] not in (1, batch)
                    or attn_mask.shape[1] not in (1, self.attn.num_attention_heads)
                    or tuple(attn_mask.shape[-2:]) != (length, length)):
                raise ValueError("Attention mask must broadcast to [B,H,J,J]")
            attn_mask = attn_mask.expand(batch, -1, -1, -1)
            blocked = blocked.unsqueeze(1)
        else:
            raise ValueError("Attention mask must be [J,J], [B,J,J], or [B,H,J,J]")
        if attn_mask.dtype == torch.bool:
            return attn_mask | blocked
        # Eager attention needs an explicit head axis for additive masks.
        merged = attn_mask.masked_fill(blocked, float("-inf"))
        return merged.unsqueeze(1) if merged.ndim == 3 else merged

    def forward(self, hidden_states, attn_mask=None, *, condition=None,
                joint_valid_mask=None, modulation=None):
        if (not torch.is_tensor(hidden_states) or hidden_states.ndim != 3
                or hidden_states.shape[-1] != self.hidden_size
                or not torch.is_floating_point(hidden_states)):
            raise ValueError("AdaLN-Zero hidden states must be floating [B,J,H]")
        shape = (*hidden_states.shape[:2], self.condition_dim)
        if (not torch.is_tensor(condition) or tuple(condition.shape) != shape
                or condition.device != hidden_states.device
                or not torch.is_floating_point(condition)):
            raise ValueError(f"AdaLN-Zero condition must be floating {shape} on the hidden-state device")
        if joint_valid_mask is None:
            valid = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        else:
            valid = torch.as_tensor(joint_valid_mask, dtype=torch.bool, device=hidden_states.device)
            if tuple(valid.shape) != tuple(hidden_states.shape[:2]):
                raise ValueError("AdaLN-Zero joint_valid_mask must be [B,J]")
        if not bool(valid.any(dim=-1).all()):
            raise ValueError("AdaLN-Zero cannot attend to an all-padding frame")
        invalid = ~valid[..., None]
        hidden_states = hidden_states.masked_fill(invalid, 0.0)
        condition = condition.masked_fill(invalid, 0.0).to(dtype=hidden_states.dtype)
        if modulation is None:
            modulation = self.adaLN_modulation(condition)
        elif tuple(modulation.shape) != (*hidden_states.shape[:2], 6 * self.hidden_size):
            raise ValueError("External decoder modulation must be [B,J,6H]")
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation.to(hidden_states.dtype).chunk(6, dim=-1)
        attn_mask = self._attention_mask(attn_mask, valid)
        bias = None if self.graph_bias is None else self.graph_bias(hidden_states.shape[1])
        attn_input = ((1 + scale_a) * self.input_layernorm(hidden_states) + shift_a).masked_fill(invalid, 0.0)
        # Flex accepts our standard 3-D bool mask; use the correct eager path for
        # additive masks or an explicit head axis instead of silently ignoring it.
        use_flex = attn_mask.dtype == torch.bool and attn_mask.ndim == 3
        attn_output, _ = self.attn(attn_input, attn_mask=attn_mask, graph_bias=bias, use_flex=use_flex)
        hidden_states = (hidden_states + gate_a * attn_output).masked_fill(invalid, 0.0)
        mlp_input = ((1 + scale_m) * self.post_attention_layernorm(hidden_states) + shift_m).masked_fill(invalid, 0.0)
        return (hidden_states + gate_m * self.mlp(mlp_input)).masked_fill(invalid, 0.0)


def process_stack(
    n_layers: int,
    in_dim: int,
    width: int,
    n_head: int,
    use_zero_init_gate: bool = False,
    min_layers: int = 0,
    bias_cfg: Optional[dict] = None,
    standard_residual: bool = False,
    condition_dim: Optional[int] = None,
) -> nn.ModuleList:
    """``n`` ProcessLayers; first layer may change dim (``in_dim -> width``)."""
    if condition_dim is not None:
        if not standard_residual:
            raise ValueError("Conditional stacks require standard_residual=True")
        if use_zero_init_gate:
            raise ValueError("AdaLN-Zero conditional stacks cannot use the zero_init_gate")
        if in_dim != width or int(condition_dim) < 1:
            raise ValueError("Conditional stacks require in_dim == width and positive condition_dim")
    n = max(int(n_layers), min_layers)
    if n <= 0:
        return nn.ModuleList()

    def make(din, dout):
        bias = ChainGraphBias(**bias_cfg) if bias_cfg else None
        if condition_dim is not None:
            return AdaLNZeroProcessLayer(din, dout, n_head, use_zero_init_gate,
                                        graph_bias=bias, condition_dim=condition_dim)
        layer_cls = StandardProcessLayer if standard_residual else ProcessLayer
        return layer_cls(din, dout, n_head, use_zero_init_gate, graph_bias=bias)

    return nn.ModuleList([make(in_dim, width), *(make(width, width) for _ in range(n - 1))])


def causal_chain_mask(
    batch: int,
    seq_len: int,
    num_joints: int,
    joint_valid_mask: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Bool mask, True = block. Joint j attends to keys <= j; invalid joints dropped as keys."""
    joint_idx = torch.arange(seq_len, device=device) % num_joints
    attn_mask = joint_idx[None, :] > joint_idx[:, None]
    if joint_valid_mask is None:
        return attn_mask.unsqueeze(0)
    if joint_valid_mask.dim() == 1:
        return (attn_mask | ~joint_valid_mask[joint_idx]).unsqueeze(0)
    if joint_valid_mask.dim() != 2:
        raise ValueError(f"joint_valid_mask must have shape [J] or [B, J], got {tuple(joint_valid_mask.shape)}")
    if joint_valid_mask.shape[0] != batch:
        raise ValueError(f"joint_valid_mask batch mismatch: got {joint_valid_mask.shape[0]}, expected {batch}")
    return attn_mask.unsqueeze(0) | (~joint_valid_mask[:, joint_idx])[:, None, :]

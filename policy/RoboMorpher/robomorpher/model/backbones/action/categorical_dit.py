# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        if temb.ndim == 3:
            # DF routes prefix/global and action/branch conditions per token.
            if temb.shape[:2] != x.shape[:2]:
                raise ValueError("Per-token AdaLN condition must match [batch, sequence]")
            scale, shift = temb.chunk(2, dim=-1)
            x = self.norm(x) * (1 + scale) + shift
        else:
            scale, shift = temb.chunk(2, dim=1)
            x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None


    def _urdf_modulation_module(self, override: Optional[nn.Module] = None) -> nn.Module:
        """Resolve the AdaLN projector for this call.

        Discrete Forcing ties the first blocks of the categorical head and the
        continuous refiner so their weights and gradients are shared. Their URDF
        AdaLN projectors must stay separate, so the owner of the call passes its
        own projector explicitly instead of relying on the attribute.
        """
        if override is not None:
            return override
        module = getattr(self, "urdf_modulation", None)
        if module is None:
            raise ValueError("URDF AdaLN requires a configured modulation and mask")
        return module

    @staticmethod
    def _reshape_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
        b, n, c = x.shape
        return x.view(b, n, heads, c // heads).transpose(1, 2).contiguous()

    @staticmethod
    def _cached_attention_mask(mask, batch, heads, q_len, k_len, dtype, device):
        if mask is None:
            return None
        if mask.ndim == 2:
            mask = mask[:, None, None, :]
        elif mask.ndim == 3:
            if mask.shape[0] == batch * heads:
                mask = mask.view(batch, heads, q_len, k_len)
            elif mask.shape[0] == batch:
                mask = mask[:, None, :, :]
            else:
                raise ValueError("Unsupported cached attention mask batch shape")
        elif mask.ndim != 4:
            raise ValueError("Unsupported cached attention mask rank")
        if mask.shape[-2:] != (q_len, k_len):
            if mask.shape[-2] == 1 and mask.shape[-1] == k_len:
                mask = mask.expand(batch, mask.shape[1], q_len, k_len)
            else:
                raise ValueError("Cached attention mask shape mismatch")
        return mask.to(device=device, dtype=dtype)

    def cache_encoder_kv(self, encoder_hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project static cross-attention keys/values once for inference KV caching."""
        attn = self.attn1
        encoder_hidden_states = encoder_hidden_states.to(dtype=attn.to_k.weight.dtype)
        if getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = self._reshape_heads(attn.to_k(encoder_hidden_states), attn.heads)
        value = self._reshape_heads(attn.to_v(encoder_hidden_states), attn.heads)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)
        return key, value

    def _attention_with_kv(self, query_states, key_value_states, *,
                           past_key_value=None, cached_key_value=None,
                           attention_mask=None):
        attn = self.attn1
        q = self._reshape_heads(attn.to_q(query_states), attn.heads)
        if getattr(attn, "norm_q", None) is not None:
            q = attn.norm_q(q)
        if cached_key_value is not None:
            new_k, new_v = cached_key_value
        else:
            if key_value_states is None:
                raise ValueError("key_value_states are required when K/V are not cached")
            key_value_states = key_value_states.to(dtype=attn.to_k.weight.dtype)
            new_k = self._reshape_heads(attn.to_k(key_value_states), attn.heads)
            new_v = self._reshape_heads(attn.to_v(key_value_states), attn.heads)
            if getattr(attn, "norm_k", None) is not None:
                new_k = attn.norm_k(new_k)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], new_k), dim=2)
            value = torch.cat((past_key_value[1], new_v), dim=2)
        else:
            key, value = new_k, new_v
        mask = self._cached_attention_mask(
            attention_mask, q.shape[0], attn.heads, q.shape[2], key.shape[2],
            q.dtype, q.device,
        )
        output = F.scaled_dot_product_attention(q, key, value, attn_mask=mask)
        output = output.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        output = attn.to_out[0](output)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output, (new_k, new_v)

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        urdf_condition: Optional[torch.Tensor] = None,
        urdf_mask: Optional[torch.Tensor] = None,
        urdf_modulation: Optional[nn.Module] = None,
    ):
        """Inference-only block forward with static cross/action KV cache."""
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)
        modulation = None
        if urdf_condition is not None:
            modulation = self._urdf_modulation_module(urdf_modulation)(urdf_condition).chunk(6, dim=-1)
            if urdf_mask is None:
                raise ValueError("URDF AdaLN requires an explicit mask")
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation
            norm_hidden_states = norm_hidden_states * (1 + scale_a) + shift_a
            norm_hidden_states = norm_hidden_states.masked_fill(~urdf_mask[..., None], 0)
        if self.pos_embed is not None:
            if position_ids is None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)
            else:
                norm_hidden_states = norm_hidden_states + self.pos_embed.pe[:, position_ids]
        if encoder_hidden_states is not None or encoder_key_value is not None:
            if encoder_key_value is None:
                attn_output, new_kv = self._attention_with_kv(
                    norm_hidden_states, encoder_hidden_states,
                    attention_mask=encoder_attention_mask)
            else:
                attn_output, new_kv = self._attention_with_kv(
                    norm_hidden_states, None, cached_key_value=encoder_key_value,
                    attention_mask=encoder_attention_mask)
        else:
            attn_output, new_kv = self._attention_with_kv(
                norm_hidden_states, norm_hidden_states,
                past_key_value=past_key_value, attention_mask=attention_mask)
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)
        if modulation is not None:
            attn_output = gate_a * attn_output
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        norm_hidden_states = self.norm3(hidden_states)
        if modulation is not None:
            norm_hidden_states = norm_hidden_states * (1 + scale_m) + shift_m
            norm_hidden_states = norm_hidden_states.masked_fill(~urdf_mask[..., None], 0)
        ff_output = self.ff(norm_hidden_states)
        if modulation is not None:
            ff_output = gate_m * ff_output
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states, new_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        urdf_condition: Optional[torch.Tensor] = None,
        urdf_mask: Optional[torch.Tensor] = None,
        urdf_modulation: Optional[nn.Module] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        modulation = None
        if urdf_condition is not None:
            modulation = self._urdf_modulation_module(urdf_modulation)(urdf_condition).chunk(6, dim=-1)
            if urdf_mask is None:
                raise ValueError("URDF AdaLN requires an explicit mask")
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation
            norm_hidden_states = norm_hidden_states * (1 + scale_a) + shift_a
            norm_hidden_states = norm_hidden_states.masked_fill(~urdf_mask[..., None], 0)
        if self.pos_embed is not None:
            if position_ids is None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)
            else:
                if position_ids.ndim != 1 or position_ids.shape[0] != norm_hidden_states.shape[1]:
                    raise ValueError("position_ids must index every token in the block sequence")
                pe = self.pos_embed.pe.to(device=norm_hidden_states.device, dtype=norm_hidden_states.dtype)
                norm_hidden_states = norm_hidden_states + pe[:, position_ids.to(norm_hidden_states.device)]

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,  # @JinhuiYE original attention_mask=attention_mask
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        if modulation is not None:
            attn_output = gate_a * attn_output
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        if modulation is not None:
            norm_hidden_states = norm_hidden_states * (1 + scale_m) + shift_m
            norm_hidden_states = norm_hidden_states.masked_fill(~urdf_mask[..., None], 0)
        ff_output = self.ff(norm_hidden_states)
        if modulation is not None:
            ff_output = gate_m * ff_output
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    # register_to_config auto-registers constructor params into config, enabling access via self.config.xxx instead of self.xxx
    @register_to_config  # Registers passed params to config. TODO: replace with our singleton pattern, implement a mergeable @merge_param_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        # This is the single forward-mode switch: canonical interleaving when true.
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self._interleave_self_attention = bool(interleave_self_attention)
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        # Timestep encoder
        #  self.config.compute_dtype may not exist, handle it in advance
        compute_dtype = getattr(self.config, "compute_dtype", torch.float32)
        self.timestep_encoder = (
            TimestepEncoder(  # TODO BUG: self.config.compute_dtype doesn't error during training but fails at eval
                embedding_dim=self.inner_dim, compute_dtype=compute_dtype
            )
        )

        all_blocks = []
        for idx in range(self.config.num_layers):

            use_self_attn = idx % 2 == 1 and self._interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D) or list of layer-wise tensors
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask=None,
        return_pre_output: bool = False,
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        is_layerwise_encoder = isinstance(encoder_hidden_states, (list, tuple))
        encoder_hidden_states = (
            [state.contiguous() for state in encoder_hidden_states]
            if is_layerwise_encoder
            else encoder_hidden_states.contiguous()
        )

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self._interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                if is_layerwise_encoder:
                    block_encoder_hidden_states = encoder_hidden_states[idx]
                else:
                    block_encoder_hidden_states = encoder_hidden_states
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=block_encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        if return_pre_output:
            if return_all_hidden_states:
                return hidden_states, all_hidden_states
            return hidden_states

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)

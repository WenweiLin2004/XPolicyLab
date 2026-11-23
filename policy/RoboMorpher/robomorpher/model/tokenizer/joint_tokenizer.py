"""Joint morphology and angle encoder used by RoboMorpher."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from robomorpher.utils.model.masking import binary_bool_mask
from .components.attention import causal_chain_mask, process_stack
from .components.embeddings import JointEmbed, AngleEmbed
from .quantizers.residual import _state_quantizer_backend


class JointTokenizer(nn.Module):
    def _width_stack(self, n: int, use_zero_init_gate: bool) -> nn.ModuleList:
        return process_stack(
            n, self.n_latent_dims, self.n_latent_dims, self.n_head, use_zero_init_gate, bias_cfg=self.bias_cfg
        )

    def _build_encoder(self, embed_hidden, quant_proj_layers, angle_mlp_hidden, use_zero_init_gate):
        del quant_proj_layers
        if not self.use_urdf_split:
            raise RuntimeError("the joint encoder requires separate morphology and angle streams")
        if self.joint_info_dim < 2:
            raise ValueError(f"urdf split requires joint_info_dim >= 2, got {self.joint_info_dim}")
        width = self.n_latent_dims
        self.urdf_embed = JointEmbed(self.urdf_dim, width, embed_hidden)
        self.angle_embed = AngleEmbed(width, angle_mlp_hidden, getattr(self, "state_input_dim", 1))
        self.urdf_proj = self._width_stack(self.urdf_layers, use_zero_init_gate)
        self.angle_proj = self._width_stack(self.angle_layers, use_zero_init_gate)
        self.state_proj = self._width_stack(self.state_layers, use_zero_init_gate)
        self.urdf_angle_film = None

    def _require_completed_ema_window(self):
        if getattr(getattr(self, "quantizer", None), "has_pending_ema", False):
            raise RuntimeError("Complete or discard the pending EMA accumulation window first")

    def _apply(self, fn, recurse=True):
        self._require_completed_ema_window()
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        self._require_completed_ema_window()
        persisted_backend = _state_quantizer_backend(state_dict)
        if persisted_backend != self.quantizer_backend:
            raise ValueError(
                f"Cannot load {persisted_backend} codebook state into {self.quantizer_backend}. "
                "Use build_tokenizer_from_config(config, state_dict=state) before loading; "
                "strict=False does not migrate codebooks."
            )
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._require_completed_ema_window()
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def quantizer_metadata(self):
        metadata = dict(self._quantizer_settings)
        if self.quantizer_backend != "vqp":
            raise ValueError("the joint tokenizer requires the vqp quantizer")
        from importlib.metadata import version

        metadata.update(
            library="vector-quantize-pytorch",
            library_version=version("vector-quantize-pytorch"),
            schema_version=1,
            padding_id=-1,
            id_schema="group_offsets_valid_only_v1",
            commitment_reduction="global_valid_token_mean_per_group",
            quantizer_precision="float32_autocast_disabled",
            ema_statistics="raw_assignment_count_and_vector_sum",
            ema_implementation="upstream_forward",
            ema_commit=("accumulation_boundary_forward"
                        if metadata["ema_update_unit"] == "accumulation_boundary" else "training_forward"),
            ema_initialization="first_nonempty_training_microbatch_kmeans",
            ema_restart_pool=("last_nonaccumulating_microbatch_valid_tokens"
                              if metadata["ema_update_unit"] == "accumulation_boundary"
                              else "current_microbatch_valid_tokens"),
            ema_on_optimizer_overflow="already_committed_in_forward_no_rollback",
        )
        return metadata

    def _valid_mask(self, x, joint_valid_mask):
        if joint_valid_mask is None:
            return None
        mask = binary_bool_mask(joint_valid_mask, device=x.device, name="joint_valid_mask")
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(x.shape[0], -1)
        if mask.shape != x.shape[:2]:
            raise ValueError(f"joint_valid_mask must match [B,J]={tuple(x.shape[:2])}, got {tuple(mask.shape)}")
        return mask

    def _attn_mask(self, x, joint_valid_mask=None):
        return causal_chain_mask(x.shape[0], x.shape[1], self.num_joints, joint_valid_mask, x.device)

    def _apply_attn_layers(self, x, layers, joint_valid_mask):
        mask = self._attn_mask(x, joint_valid_mask)
        for layer in layers:
            x = layer(x, attn_mask=mask)
        return x

    def encode_urdf(self, urdf: torch.Tensor, joint_valid_mask: Optional[torch.Tensor] = None):
        """urdf (B, J, urdf_dim) -> h_urdf (B, J, n_latent_dims). Angle-invariant."""
        if not self.use_urdf_split:
            raise RuntimeError("encode_urdf requires encoder.urdf_layers > 0")
        if urdf.shape[-1] != self.urdf_dim:
            raise ValueError(f"urdf last dim must be {self.urdf_dim}, got {tuple(urdf.shape)}")
        joint_valid_mask = self._valid_mask(urdf, joint_valid_mask)
        if joint_valid_mask is not None:
            urdf = urdf.masked_fill(~joint_valid_mask[..., None], 0.0)
        hidden = self._apply_attn_layers(self.urdf_embed(urdf), self.urdf_proj, joint_valid_mask)
        if joint_valid_mask is not None:
            hidden = hidden.masked_fill(~joint_valid_mask[..., None], 0.0)
        return hidden

    def encode_angle(self, angle: torch.Tensor, joint_valid_mask: Optional[torch.Tensor] = None):
        """angle (B, J, 1) -> h_angle (B, J, n_latent_dims). URDF-invariant."""
        if not self.use_urdf_split:
            raise RuntimeError("encode_angle requires encoder.urdf_layers > 0")
        if angle.shape[-1] != 1:
            raise ValueError(f"angle last dim must be 1, got {tuple(angle.shape)}")
        joint_valid_mask = self._valid_mask(angle, joint_valid_mask)
        if joint_valid_mask is not None:
            angle = angle.masked_fill(~joint_valid_mask[..., None], 0.0)
        hidden = self._apply_attn_layers(self.angle_embed(angle), self.angle_proj, joint_valid_mask)
        if joint_valid_mask is not None:
            hidden = hidden.masked_fill(~joint_valid_mask[..., None], 0.0)
        return hidden

    def _decoder_input(self, z):
        # Frozen tokenizers can store decoder weights in BF16 while codebooks stay FP32.
        return z.to(dtype=next(self.decoder_head.parameters()).dtype)

    def _split_output(self, out):
        return out[..., : self.ee_dim], out[..., self.ee_dim : self.ee_dim + self.joint_dim]

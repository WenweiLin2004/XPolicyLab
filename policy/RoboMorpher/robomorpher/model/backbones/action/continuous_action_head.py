"""StarVLA/DF continuous flow-matching action head.

The continuous branch receives only the decoded physical source action.
Under the released Discrete Forcing branch-isolation contract, discrete RVQ ID
sequences are removed before the shared trunk; the source is GT-token-decoded
in training and final predicted-token decoding at inference. The per-slot valid
mask is projected as an embedding, never appended as a sequence token. Every
padded action scalar and time slot is masked before attention, noise, loss and
output.
"""
from __future__ import annotations
from collections.abc import Mapping
from functools import partial
import math
import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F
from robomorpher.utils.model.masking import binary_bool_mask
from robomorpher.model.backbones.action.scheduler import ActionScheduler


class _SinusoidalActionTime(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        if int(embedding_dim) % 2:
            raise ValueError("QwenPI action time embedding requires an even hidden dimension")
        self.embedding_dim = int(embedding_dim)

    def forward(self, timesteps):
        if timesteps.ndim != 1:
            raise ValueError("action timestep must be [B]")
        half = self.embedding_dim // 2
        freq = torch.arange(half, device=timesteps.device, dtype=torch.float32)
        freq = torch.exp(-freq * (math.log(10000.0) / half))
        phase = timesteps.float()[:, None] * freq[None]
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


class _QwenPIActionEncoder(nn.Module):
    """QwenPI-v3's per-timestep action MLP plus a shared flow-time code."""
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_dim)
        self.layer2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.time = _SinusoidalActionTime(hidden_dim)

    def forward(self, actions, timesteps):
        time = self.time(timesteps).to(dtype=actions.dtype)[:, None, :].expand(-1, actions.shape[1], -1)
        x = torch.cat([self.layer1(actions), time], dim=-1)
        x = self.layer2(x)
        x = x * torch.sigmoid(x)
        return self.layer3(x)


class _QwenPIActionDecoder(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, 1024)
        self.layer2 = nn.Linear(1024, action_dim)

    def forward(self, hidden):
        return self.layer2(torch.relu(self.layer1(hidden)))


class ContinuousActionHead(nn.Module):
    def __init__(self, *, action_dim: int, action_horizon: int,
                 vl_hidden_dim: int, num_layers: int, hidden_dim: int,
                 num_heads: int, state_dim: int = 0, future_tokens: int = 32,
                 max_seq_len: int | None = None, add_pos_embed: bool = True,
                 dropout: float = 0.0, diffusion_model_cfg: Mapping | None = None,
                 loss_type: str = "l2", num_inference_steps: int = 10,
                 flow_matching_style: str = "starvla",
                 noise_beta_alpha: float = 1.5, noise_beta_beta: float = 1.0,
                 noise_s: float = 0.999, num_timestep_buckets: int = 1000,
                 one_step_loss_prob: float = 0.0,
                 urdf_feature_dim: int = 0, urdf_conditioning: bool = False,
                 urdf_arm_count: int = 0, urdf_joints_per_arm: int = 0,
                 valid_mask_embedding: bool = True,
                 discrete_mode: str = "categorical"):
        super().__init__()
        if discrete_mode not in {"categorical", "joint_maskgit"}:
            raise ValueError("discrete_mode must be categorical or joint_maskgit")
        self.discrete_mode = discrete_mode
        if any(int(v) < 1 for v in (action_dim, action_horizon, num_layers, hidden_dim, num_heads)):
            raise ValueError("Continuous dimensions must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if state_dim < 0 or future_tokens < 0:
            raise ValueError("state_dim and future_tokens must be nonnegative")
        if loss_type not in {"l1", "l2"}:
            raise ValueError("continuous_loss must be l1 or l2")
        if int(num_inference_steps) < 1:
            raise ValueError("continuous inference steps must be positive")
        flow_matching_style = str(flow_matching_style).lower()
        if flow_matching_style not in {"starvla", "shifted_time"}:
            raise ValueError("continuous_flow_matching must be starvla or shifted_time")
        if int(num_timestep_buckets) < 2:
            raise ValueError("num_timestep_buckets must be >= 2")
        if isinstance(urdf_feature_dim, bool) or int(urdf_feature_dim) < 0:
            raise ValueError("urdf_feature_dim must be a nonnegative integer")
        if urdf_conditioning:
            if int(urdf_feature_dim) < 1:
                raise ValueError("urdf_conditioning requires a positive urdf_feature_dim")
            if int(urdf_arm_count) < 1 or int(urdf_joints_per_arm) < 1:
                raise ValueError(
                    "urdf_conditioning requires urdf_arm_count and urdf_joints_per_arm "
                    "so the two arms can be conditioned separately")
        for value, name in ((noise_beta_alpha, "noise_beta_alpha"), (noise_beta_beta, "noise_beta_beta"), (noise_s, "noise_s")):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if float(noise_s) > 1:
            raise ValueError("noise_s must be <= 1")
        if not math.isfinite(float(one_step_loss_prob)) or not 0.0 <= float(one_step_loss_prob) <= 1.0:
            raise ValueError("one_step_loss_prob must be finite in [0,1]")
        self.action_dim, self.action_horizon = int(action_dim), int(action_horizon)
        self.num_layers, self.hidden_dim = int(num_layers), int(hidden_dim)
        self.vl_hidden_dim, self.state_dim = int(vl_hidden_dim), int(state_dim)
        self.loss_type, self.num_inference_steps = str(loss_type), int(num_inference_steps)
        self.flow_matching_style = flow_matching_style
        self.noise_beta_alpha, self.noise_beta_beta = float(noise_beta_alpha), float(noise_beta_beta)
        self.noise_s, self.num_timestep_buckets = float(noise_s), int(num_timestep_buckets)
        self.one_step_loss_prob = float(one_step_loss_prob)
        self.add_pos_embed = bool(add_pos_embed)
        self.urdf_feature_dim = int(urdf_feature_dim)
        self.urdf_conditioning = bool(urdf_conditioning)
        self.urdf_arm_count = int(urdf_arm_count)
        self.urdf_joints_per_arm = int(urdf_joints_per_arm)
        self.valid_mask_embedding_enabled = bool(valid_mask_embedding)
        # A per-slot validity projection tells the action head whether a zero
        # value is a real joint angle or a padded 6-DoF slot. It is a condition
        # only: valid_mask still excludes padding from noise, loss, and output.
        self.valid_mask_embedding = (
            nn.Parameter(torch.empty(self.action_dim, self.hidden_dim))
            if self.valid_mask_embedding_enabled else None
        )
        if self.valid_mask_embedding is not None:
            nn.init.normal_(self.valid_mask_embedding, std=0.02)
        # Pooling keeps the arms separate, so the AdaLN width is arms x features.
        self.urdf_condition_width = self.urdf_feature_dim * max(1, self.urdf_arm_count)
        dit_cfg = dict(diffusion_model_cfg or {})
        allowed = {"num_layers", "attention_head_dim", "cross_attention_dim", "output_dim",
                   "dropout", "final_dropout", "interleave_self_attention",
                   "norm_type", "positional_embeddings",
                   "max_num_positional_embeddings", "attention_bias", "activation_fn",
                   "upcast_attention", "norm_elementwise_affine", "norm_eps"}
        unknown = set(dit_cfg) - allowed
        if unknown:
            raise ValueError(f"Unsupported diffusion_model_cfg settings: {sorted(unknown)}")
        head_dim = dit_cfg.pop("attention_head_dim", None)
        if head_dim is not None and int(head_dim) != hidden_dim // num_heads:
            raise ValueError("continuous attention_head_dim conflicts with hidden_dim / num_heads")
        for key, value in (("num_layers", num_layers), ("cross_attention_dim", vl_hidden_dim), ("output_dim", action_dim)):
            configured = dit_cfg.pop(key, value)
            if int(configured) != int(value):
                raise ValueError(f"diffusion_model_cfg.{key} conflicts with continuous dimensions")
        nested_dropout = dit_cfg.pop("dropout", None)
        if nested_dropout is not None and float(nested_dropout) != float(dropout):
            raise ValueError("continuous dropout conflicts with diffusion_model_cfg.dropout")
        dropout = float(dropout if nested_dropout is None else nested_dropout)
        if not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be finite in [0,1]")
        interleave = dit_cfg.pop("interleave_self_attention", None)
        self.interleave_self_attention = bool(interleave if interleave is not None else True)
        # starVLA and DF both leave block sinusoids off. Action tokens keep the
        # learned position table added before the blocks.
        positional = dit_cfg.pop("positional_embeddings", None)
        if positional not in {None, "sinusoidal"}:
            raise ValueError("positional_embeddings must be sinusoidal or null")
        if discrete_mode == "joint_maskgit" and positional is not None:
            raise ValueError("joint_maskgit follows DF: block positional_embeddings must be null")
        norm_type = dit_cfg.pop("norm_type", "ada_norm")
        if norm_type not in {"ada_norm", "layer_norm"}:
            raise ValueError("continuous norm_type must be ada_norm or layer_norm")
        final_dropout = bool(dit_cfg.pop("final_dropout", bool(dropout)))
        prefix_len = int(future_tokens) + (1 if state_dim else 0)
        # QwenPI_v3 uses one action token per timestep whose feature vector
        # contains all action dimensions.  Keep the same temporal tokenization;
        # padded dimensions are masked only in the feature values and loss.
        self.seq_len = int(action_horizon)
        capacity = int(max_seq_len or (self.seq_len + prefix_len))
        if capacity < self.seq_len + prefix_len:
            raise ValueError(f"max_seq_len {capacity} < required {self.seq_len + prefix_len}")
        from robomorpher.model.backbones.action.categorical_dit import DiT
        self.model = DiT(
            num_layers=num_layers, num_attention_heads=num_heads,
            attention_head_dim=hidden_dim // num_heads, output_dim=hidden_dim,
            cross_attention_dim=vl_hidden_dim, norm_type=norm_type,
            dropout=dropout, final_dropout=final_dropout,
            interleave_self_attention=self.interleave_self_attention,
            positional_embeddings=positional,
            max_num_positional_embeddings=capacity, **dit_cfg,
        )
        # This head owns one URDF AdaLN projector per block. Shared DF blocks are
        # the categorical head's module objects, so the projector is handed to the
        # block explicitly and never stored on the block itself.
        self.urdf_projector = None
        self.urdf_modulations = nn.ModuleList()
        if self.urdf_conditioning:
            for _ in range(self.num_layers):
                modulation = nn.Sequential(
                    nn.SiLU(), nn.Linear(self.urdf_condition_width, 6 * hidden_dim))
                nn.init.zeros_(modulation[-1].weight)
                nn.init.zeros_(modulation[-1].bias)
                self.urdf_modulations.append(modulation)
        self.action_encoder = _QwenPIActionEncoder(action_dim, hidden_dim)
        self.action_decoder = _QwenPIActionDecoder(hidden_dim, action_dim)
        self.position_embedding = nn.Embedding(capacity, hidden_dim) if add_pos_embed else None
        self.future_tokens = nn.Parameter(torch.empty(int(future_tokens), hidden_dim))
        self.state_encoder = (nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)) if state_dim else None)
        self.beta_dist = Beta(self.noise_beta_alpha, self.noise_beta_beta)
        self.scheduler = None
        if self.flow_matching_style == "shifted_time":
            self.scheduler = ActionScheduler()
            self.scheduler.set_timesteps(ActionScheduler.num_train_timesteps, training=True)
        nn.init.normal_(self.action_encoder.layer1.weight, std=0.02)
        nn.init.zeros_(self.action_encoder.layer1.bias)
        nn.init.normal_(self.future_tokens, std=0.02)
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight, std=0.02)

    @staticmethod
    def _key_mask(valid, dtype):
        return torch.zeros(valid.shape[0], 1, valid.shape[1], device=valid.device, dtype=dtype).masked_fill(
            ~valid[:, None, :], torch.finfo(dtype).min)

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing = False

    def _check(self, actions, valid_mask):
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (self.action_horizon, self.action_dim):
            raise ValueError("continuous actions must be [B, action_horizon, action_dim]")
        if valid_mask.shape != actions.shape:
            raise ValueError("continuous valid_mask must match actions")
        valid_mask = binary_bool_mask(valid_mask, device=actions.device, name="continuous_action_valid_mask")
        if not bool(torch.isfinite(actions[valid_mask]).all()):
            raise ValueError("Nonfinite valid continuous action target")
        if not bool(valid_mask.any()):
            raise ValueError("Continuous action batch has no valid dimensions")
        if self.discrete_mode == "joint_maskgit" and not bool(valid_mask.flatten(1).any(-1).all()):
            raise ValueError("Every continuous action sample must contain valid dimensions")
        return valid_mask

    def _hidden(self, actions, valid_mask, timestep, state, vl_attention_mask,
                urdf_condition=None):
        if self.discrete_mode == "joint_maskgit" and not getattr(self.model, "uses_df_token_routing", False):
            raise RuntimeError("joint_maskgit continuous head requires the DF per-token Hybrid owner")
        batch, device = actions.shape[0], actions.device
        if isinstance(state, Mapping):
            # A mapping may carry a plain continuous state alongside optional
            # URDF features. Only a head explicitly built with URDF AdaLN may
            # promote the mapping to urdf_condition.
            if urdf_condition is None and self.urdf_conditioning:
                urdf_condition = state
            state = state.get("state")
        if urdf_condition is not None:
            if not self.urdf_conditioning:
                raise ValueError("URDF condition supplied to a continuous head without URDF AdaLN")
            if not isinstance(urdf_condition, Mapping):
                raise ValueError("URDF condition must contain urdf_features and urdf_valid_mask")
            features = urdf_condition.get("urdf_features")
            uv = urdf_condition.get("urdf_valid_mask")
            if not torch.is_tensor(features) or not torch.is_tensor(uv):
                raise ValueError("URDF condition must contain tensor features and mask")
            uv = binary_bool_mask(uv, device=device, name="urdf_valid_mask")
            if features.ndim != 3 or features.shape[0] != batch or features.shape[1] != uv.shape[1] or features.shape[-1] != self.urdf_feature_dim:
                raise ValueError("URDF feature shape does not match continuous AdaLN contract")
            features = torch.where(uv[..., None], features.to(dtype=actions.dtype), torch.zeros_like(features, dtype=actions.dtype))
            if not bool(torch.isfinite(features).all()):
                raise ValueError("Nonfinite valid URDF feature")
            # Pool each arm separately: a single pooled vector would make the
            # conditioning invariant to swapping the two arms, which is wrong
            # whenever the left and right embodiments differ.
            arms, joints = self.urdf_arm_count, self.urdf_joints_per_arm
            if features.shape[1] != arms * joints:
                raise ValueError("URDF token count does not match the arm geometry contract")
            per_arm = features.reshape(batch, arms, joints, self.urdf_feature_dim)
            valid = uv.reshape(batch, arms, joints)
            # A single-arm sample has an entirely absent, zero-masked arm.
            # Pooling below leaves that arm at zero; only an empty sample is invalid.
            if not bool(valid.any(dim=-1).any(dim=-1).all()):
                raise ValueError("Every sample needs at least one valid URDF joint")
            pooled = per_arm.sum(2) / valid.sum(2, keepdim=True).clamp_min(1).to(per_arm.dtype)
            urdf_vector = pooled.reshape(batch, arms * self.urdf_feature_dim)
        else:
            urdf_vector = None
        time_valid = valid_mask.any(dim=-1)
        clean_actions = actions.masked_fill(~valid_mask, 0)
        hidden = self.action_encoder(clean_actions, timestep)
        if self.valid_mask_embedding is not None:
            # [B,H,D] @ [D,C] -> [B,H,C]; mask is boolean and receives no grad.
            # FSDP may expose an unsharded FlatParameter view as a 1-D tensor
            # when this parameter is accessed directly. Restore its logical
            # [action_dim, hidden_dim] shape before the matmul.
            mask_embedding = self.valid_mask_embedding
            expected = self.action_dim * self.hidden_dim
            # Depending on FSDP wrapping, a flattened parameter can appear as
            # [N], [1, N], or another view with the same number of elements.
            # Use numel rather than ndim so all of these views recover the
            # logical [action_dim, hidden_dim] contract.
            if tuple(mask_embedding.shape) != (self.action_dim, self.hidden_dim):
                if mask_embedding.numel() != expected:
                    raise RuntimeError(
                        "FSDP-flattened valid_mask_embedding has unexpected size "
                        f"{mask_embedding.numel()} (expected {expected})")
                mask_embedding = mask_embedding.reshape(self.action_dim, self.hidden_dim)
            mask_condition = torch.matmul(
                valid_mask.to(dtype=hidden.dtype), mask_embedding.to(dtype=hidden.dtype)
            )
            hidden = hidden + mask_condition
        hidden = hidden.masked_fill(~time_valid[..., None], 0)
        # Strict DF branch isolation: only the decoded physical action sequence
        # enters the continuous path. RVQ IDs are never extra sequence tokens.
        action_hidden, action_valid = hidden, time_valid
        shared_positions = bool(getattr(self.model, "uses_df_token_routing", False))
        if shared_positions:
            positions = self.model.action_positions("continuous", action_hidden.shape[1], device)
            action_hidden = action_hidden + self.model.shared_position_embedding(positions)[None]
        elif self.position_embedding is not None:
            if action_hidden.shape[1] > self.position_embedding.num_embeddings:
                raise ValueError("continuous action position embedding capacity is too small")
            action_hidden = action_hidden + self.position_embedding(
                torch.arange(action_hidden.shape[1], device=device))[None]
        action_hidden = action_hidden.masked_fill(~action_valid[..., None], 0)
        pieces, masks = [], []
        if state is not None:
            if self.state_encoder is None or not torch.is_tensor(state) or state.shape != (batch, self.state_dim):
                raise ValueError("continuous state condition has the wrong shape")
            pieces.append(self.state_encoder(state.to(dtype=actions.dtype)).unsqueeze(1))
            masks.append(torch.ones(batch, 1, dtype=torch.bool, device=device))
        # FSDP can expose a direct Parameter as a flattened 1-D view while
        # wrapping the parent head. Restore the logical [N,D] future-token
        # layout before expanding across the batch.
        future_tokens = (self.model.shared_future_tokens if shared_positions else self.future_tokens)
        # Recover [num_future_tokens, hidden_dim] from any FSDP flattened view
        # such as [numel] or [1, numel].  A normal unflattened parameter is
        # left untouched.
        if future_tokens.ndim != 2 or future_tokens.shape[-1] != self.hidden_dim:
            if future_tokens.numel() % self.hidden_dim:
                raise RuntimeError(
                    "FSDP-flattened future_tokens is not divisible by hidden_dim")
            future_tokens = future_tokens.reshape(-1, self.hidden_dim)
        if future_tokens.shape[0]:
            pieces.append(future_tokens[None].expand(batch, -1, -1))
            masks.append(torch.ones(batch, future_tokens.shape[0], dtype=torch.bool, device=device))
        pieces.append(action_hidden); masks.append(action_valid)
        full_valid = torch.cat(masks, dim=1)
        hidden = torch.cat(pieces, dim=1).masked_fill(~full_valid[..., None], 0)
        urdf_cond_full = None
        if urdf_vector is not None:
            urdf_cond_full = urdf_vector[:, None, :].expand(-1, hidden.shape[1], -1)
            urdf_cond_full = urdf_cond_full.to(hidden.dtype).masked_fill(~full_valid[..., None], 0)
        if vl_attention_mask is None:
            raise ValueError("continuous action head requires a VLM attention mask")
        valid_vl = binary_bool_mask(vl_attention_mask, device=device, name="vl_attention_mask")
        if valid_vl.ndim != 2 or valid_vl.shape[0] != batch or not valid_vl.any(-1).all():
            raise ValueError("VLM attention mask is empty or has the wrong shape")
        return hidden, full_valid, self._key_mask(full_valid, hidden.dtype), self._key_mask(valid_vl, hidden.dtype), urdf_cond_full

    def _forward_raw(self, vl_embs_list, actions, valid_mask, timestep, state=None,
                     vl_attention_mask=None, urdf_condition=None, *,
                     global_timestep=None, continuous_timestep=None,
                     discrete_timestep=None):
        hidden, full_valid, self_mask, cross_mask, urdf_cond_full = self._hidden(
            actions, valid_mask, timestep, state, vl_attention_mask,
            urdf_condition)
        if global_timestep is None:
            global_timestep = timestep
        if continuous_timestep is None:
            continuous_timestep = timestep
        if discrete_timestep is None:
            discrete_timestep = torch.full_like(
                timestep, self.num_timestep_buckets - 1)
        if hasattr(self.model, "forward_branch"):
            hidden = self.model.forward_branch(
                "continuous", hidden, vl_embs_list, full_valid=full_valid,
                self_attention_mask=self_mask, cross_attention_mask=cross_mask,
                global_timestep=global_timestep,
                continuous_timestep=continuous_timestep,
                discrete_timestep=discrete_timestep,
                urdf_condition=urdf_cond_full, urdf_mask=full_valid,
                urdf_modulations=(self.urdf_modulations
                                  if len(self.urdf_modulations) else None),
                **({"prefix_len": hidden.shape[1] - self.seq_len}
                   if getattr(self.model, "uses_df_token_routing", False) else {}),
            )
            output = self.action_decoder(hidden[:, -self.seq_len:, :])
            return output.masked_fill(~valid_mask, 0)
        temb = self.model.timestep_encoder(global_timestep.to(device=actions.device))
        layerwise = isinstance(vl_embs_list, (list, tuple))
        if layerwise:
            if len(vl_embs_list) != self.num_layers:
                raise ValueError("VLM layer count does not match continuous head")
            enc = [None if h is None else h.contiguous() for h in vl_embs_list]
        else:
            enc = vl_embs_list.contiguous()
        for idx, block in enumerate(self.model.transformer_blocks):
            is_self = self.interleave_self_attention and idx % 2 == 1
            block_enc = None if is_self else (enc[idx] if layerwise else enc)
            kwargs = dict(encoder_hidden_states=block_enc,
                          encoder_attention_mask=self_mask if is_self else cross_mask,
                          temb=temb)
            if urdf_cond_full is not None:
                kwargs.update(urdf_condition=urdf_cond_full, urdf_mask=full_valid,
                              urdf_modulation=self.urdf_modulations[idx])
            if self.model.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    partial(block, **kwargs), hidden, use_reentrant=False)
            else:
                hidden = block(hidden, **kwargs)
            hidden = hidden.masked_fill(~full_valid[..., None], 0)
        # starVLA LayerwiseFM trains and samples with return_pre_output=True.
        # Block AdaLN stays. The DiT's final AdaLN and output projection do not.
        output = self.action_decoder(hidden[:, -self.seq_len:, :])
        return output.masked_fill(~valid_mask, 0)

    def _forward_df(self, vl_embs_list, actions, valid_mask, timestep,
                    state=None, vl_attention_mask=None, urdf_condition=None,
                    *, global_timestep=None, continuous_timestep=None,
                    discrete_timestep=None):
        """Call the three-time-stream path only for a Hybrid DiT.

        A plain DiT is the starVLA head: blocks, then the action decoder.
        """
        model = getattr(self, "model", None)
        if hasattr(model, "forward_branch"):
            return self._forward_raw(
                vl_embs_list, actions, valid_mask, timestep, state,
                vl_attention_mask, urdf_condition,
                global_timestep=global_timestep,
                continuous_timestep=continuous_timestep,
                discrete_timestep=discrete_timestep,
            )
        return self._forward_raw(
            vl_embs_list, actions, valid_mask, timestep, state,
            vl_attention_mask, urdf_condition,
        )

    @staticmethod
    def _masked_error_rows(prediction, target, valid_mask, loss_type):
        """Per-row mean error over the valid coordinates only."""
        error = ((prediction - target).abs() if loss_type == "l1" else (prediction - target).square())
        error = error.masked_fill(~valid_mask, 0)
        counts = valid_mask.flatten(1).sum(1).clamp_min(1)
        return error.flatten(1).sum(1) / counts

    @classmethod
    def _masked_error(cls, prediction, target, valid_mask, loss_type):
        return cls._masked_error_rows(prediction, target, valid_mask, loss_type).mean()

    def refinement_loss(self, vl_embs_list, source, target, valid_mask, *, state=None, vl_attention_mask=None,
                        urdf_condition=None, alpha=0.3, one_step_loss_weight=0.0, one_step_loss_prob=None,
                        phase_timestep=None):
        """Discrete-Forcing continuous refinement objective.

        ``source`` is the detached hard-decoded discrete action in normalized
        physical coordinates. The source is mixed with noise once, then the
        continuous branch learns a flow from that noisy source to the target.
        A one-step reconstruction term follows the DF paper and never sends
        gradients into the discrete source.
        """
        valid_mask = self._check(source, valid_mask)
        if target.shape != source.shape:
            raise ValueError("DF source and target shapes must match")
        if not bool(torch.isfinite(target[valid_mask]).all()):
            raise ValueError("Nonfinite continuous refinement target")
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("DF alpha must be in (0,1]")
        if float(one_step_loss_weight) < 0 or not math.isfinite(float(one_step_loss_weight)):
            raise ValueError("DF one_step_loss_weight must be finite and nonnegative")
        prob = self.one_step_loss_prob if one_step_loss_prob is None else float(one_step_loss_prob)
        if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
            raise ValueError("DF one_step_loss_prob must be finite in [0,1]")
        # Official DF keeps the continuous trajectory in the VLM hidden dtype.
        compute_dtype = vl_embs_list[0].dtype
        source = source.detach().to(dtype=compute_dtype).masked_fill(~valid_mask, 0)
        target = target.detach().to(dtype=compute_dtype).masked_fill(~valid_mask, 0)
        noise = torch.randn(source.shape, device=source.device, dtype=compute_dtype).masked_fill(~valid_mask, 0)
        mixed_source = (float(alpha) * source + (1.0 - float(alpha)) * noise).masked_fill(~valid_mask, 0)
        # Discrete Forcing time convention: gamma=0 is the source end and
        # gamma=1 the target end, so the vector field is v = target - source.
        # The one-step term therefore reads the source at exactly the same time
        # index (0) that the flow term uses at gamma=0.
        joint_mode = self.discrete_mode == "joint_maskgit"
        if phase_timestep is None and joint_mode:
            # DF continuous half of sample_time; keep schedule arithmetic FP32.
            beta = self.beta_dist.sample((source.shape[0],)).to(device=source.device, dtype=torch.float32)
            phase_timestep = ((self.noise_s - 0.5 * beta) / self.noise_s).clamp(0.5, 1.0)
        if phase_timestep is None:
            phase_timestep = None
            gamma = torch.rand(source.shape[0], device=source.device, dtype=source.dtype).clamp_(1e-4, 1 - 1e-4)
        else:
            phase_timestep = phase_timestep.to(device=source.device, dtype=torch.float32 if joint_mode else source.dtype)
            if phase_timestep.shape != (source.shape[0],) or not bool(torch.isfinite(phase_timestep).all()) or bool((phase_timestep < 0.5).any()) or bool((phase_timestep > 1.0).any()):
                raise ValueError("continuous phase_timestep must be finite in [0.5, 1] with shape [B]")
            gamma = ((phase_timestep - 0.5) * 2.0).clamp_(0.0 if joint_mode else 1e-4, 1.0 if joint_mode else 1 - 1e-4)
        flow_gamma = gamma.to(compute_dtype) if joint_mode else gamma
        flow_input = ((1 - flow_gamma[:, None, None]) * mixed_source + flow_gamma[:, None, None] * target).masked_fill(~valid_mask, 0)
        timestep = (gamma * self.num_timestep_buckets).long().clamp_(0, self.num_timestep_buckets - 1)
        # DF has three independent time streams. The shared/global stream
        # receives the absolute phase t in [.5,1], the continuous branch
        # receives the re-scaled gamma in [0,1], and the discrete branch is
        # held at its fully resolved endpoint.
        global_timestep = (
            ((phase_timestep if phase_timestep is not None else
              (0.5 + 0.5 * gamma)) * self.num_timestep_buckets)
            .long().clamp_(0, self.num_timestep_buckets - 1)
        )
        discrete_timestep = torch.full_like(timestep, self.num_timestep_buckets - 1)
        prediction = self._forward_df(
            vl_embs_list, flow_input, valid_mask, timestep, state,
            vl_attention_mask, urdf_condition,
            global_timestep=global_timestep,
            continuous_timestep=timestep,
            discrete_timestep=discrete_timestep,
        )
        flow_target = (target - mixed_source).masked_fill(~valid_mask, 0)
        flow_loss = self._masked_error(prediction.float() if joint_mode else prediction,
                                       flow_target.float() if joint_mode else flow_target,
                                       valid_mask, self.loss_type)
        if prob <= 0.0 or one_step_loss_weight <= 0.0:
            # Discrete Forcing draws the one-step term on a subset of the
            # continuous rows; keep the same stochastic selection semantics.
            one_step_loss = flow_loss.new_zeros(())
        elif joint_mode:
            # DF selects first, so probability < 1 also saves endpoint compute.
            selected = torch.where(torch.rand(source.shape[0], device=source.device) < prob)[0]
            local_has_selected = selected.numel() > 0
            any_has_selected = local_has_selected
            if (torch.distributed.is_available() and torch.distributed.is_initialized()
                    and torch.distributed.get_world_size() > 1):
                # Nested FSDP blocks must enter the same forward/backward
                # collectives on every rank, including ranks with no selected row.
                # Keep the flag on the input device so NCCL never sees a CPU tensor.
                any_selected = torch.tensor(int(local_has_selected), device=source.device, dtype=torch.int32)
                torch.distributed.all_reduce(any_selected, op=torch.distributed.ReduceOp.MAX)
                any_has_selected = bool(any_selected.item())
            if not any_has_selected:
                one_step_loss = flow_loss.new_zeros(())
            else:
                if not local_has_selected:
                    selected = torch.zeros(1, device=source.device, dtype=torch.long)

                def take(value):
                    if torch.is_tensor(value):
                        return value.index_select(0, selected)
                    if isinstance(value, Mapping):
                        return {key: take(item) for key, item in value.items()}
                    if isinstance(value, (list, tuple)):
                        return type(value)(take(item) for item in value)
                    return value

                zero_t = torch.zeros(selected.numel(), device=source.device, dtype=torch.long)
                selected_source = mixed_source.index_select(0, selected)
                selected_valid = valid_mask.index_select(0, selected)
                one_step_prediction = self._forward_df(
                    take(vl_embs_list), selected_source, selected_valid, zero_t,
                    take(state), take(vl_attention_mask), take(urdf_condition),
                    global_timestep=torch.full_like(zero_t, self.num_timestep_buckets // 2),
                    continuous_timestep=zero_t,
                    discrete_timestep=torch.full_like(zero_t, self.num_timestep_buckets - 1),
                )
                one_step_action = (selected_source + one_step_prediction).masked_fill(~selected_valid, 0)
                rows = self._masked_error_rows(one_step_action.float(), target.index_select(0, selected).float(),
                                               selected_valid, self.loss_type)
                # Unselected rows contribute zero; normalize over ALL rows.
                # The dummy rank must retain its zero-valued autograd path so
                # FSDP backward also participates, without adding an objective.
                one_step_loss = rows.sum() / source.shape[0] if local_has_selected else rows.sum() * 0.0
        else:
            zero_t = torch.zeros(source.shape[0], device=source.device, dtype=torch.long)
            half_t = torch.full_like(zero_t, self.num_timestep_buckets // 2)
            discrete_t = torch.full_like(zero_t, self.num_timestep_buckets - 1)
            one_step_prediction = self._forward_df(
                vl_embs_list, mixed_source, valid_mask, zero_t, state,
                vl_attention_mask, urdf_condition,
                global_timestep=half_t,
                continuous_timestep=zero_t,
                discrete_timestep=discrete_t,
            )
            one_step_action = (mixed_source + one_step_prediction).masked_fill(~valid_mask, 0)
            # Official DF squares this reconstruction error in float32.
            rows = self._masked_error_rows(one_step_action.float(), target.float(), valid_mask, self.loss_type)
            if prob < 1.0:
                keep = (torch.rand(source.shape[0], device=source.device) < prob).to(rows.dtype)
                rows = rows * keep
            one_step_loss = rows.mean()
        total = flow_loss + float(one_step_loss_weight) * one_step_loss
        return {"continuous_loss": total, "continuous_flow_loss": flow_loss,
                "continuous_one_step_loss": one_step_loss, "loss_mask": valid_mask}

    @torch.inference_mode()
    def predict_from_source(self, vl_embs_list, source, valid_mask, *, state=None,
                            vl_attention_mask=None, urdf_condition=None,
                            num_inference_steps=None):
        """Refine a decoded discrete action with Euler steps from 0 to 1.

        One step is the t=0 correction: source + velocity. More steps split
        that same integral. The source already includes the training noise mix.
        """
        valid_mask = self._check(source, valid_mask)
        source = source.masked_fill(~valid_mask, 0)
        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        if steps < 1:
            raise ValueError("continuous inference steps must be positive")
        sample = source
        dt = 1.0 / steps
        for step in range(steps):
            timestep_id = min(int(step / steps * self.num_timestep_buckets),
                              self.num_timestep_buckets - 1)
            timestep = torch.full((source.shape[0],), timestep_id,
                                  device=source.device, dtype=torch.long)
            global_timestep = torch.full(
                (source.shape[0],), self.num_timestep_buckets // 2 +
                int(step / steps * (self.num_timestep_buckets // 2)),
                device=source.device, dtype=torch.long)
            discrete_timestep = torch.full_like(timestep, self.num_timestep_buckets - 1)
            velocity = self._forward_df(
                vl_embs_list, sample, valid_mask, timestep, state,
                vl_attention_mask, urdf_condition,
                global_timestep=global_timestep,
                continuous_timestep=timestep,
                discrete_timestep=discrete_timestep,
            )
            sample = (sample + dt * velocity).masked_fill(~valid_mask, 0)
        return sample

    def forward(self, vl_embs_list, actions, valid_mask, state=None, vl_attention_mask=None, urdf_condition=None):
        valid_mask = self._check(actions, valid_mask)
        batch = actions.shape[0]
        clean = actions.masked_fill(~valid_mask, 0)
        noise = torch.randn_like(clean).masked_fill(~valid_mask, 0)
        if self.flow_matching_style == "starvla":
            sample = self.beta_dist.sample((batch,)).to(device=actions.device, dtype=actions.dtype)
            t = ((self.noise_s - sample) / self.noise_s)[:, None, None]
            noisy = ((1 - t) * noise + t * clean).masked_fill(~valid_mask, 0)
            timestep = (t[:, 0, 0] * self.num_timestep_buckets).long()
            prediction = self._forward_raw(vl_embs_list, noisy, valid_mask, timestep, state, vl_attention_mask, urdf_condition)
            target = (clean - noise).masked_fill(~valid_mask, 0)
            error = ((prediction - target).abs() if self.loss_type == "l1" else (prediction - target).square()).masked_fill(~valid_mask, 0)
            counts = valid_mask.flatten(1).sum(1).clamp_min(1)
            loss = (error.flatten(1).sum(1) / counts).mean()
        else:
            ids = torch.randint(0, self.scheduler.num_train_timesteps, (batch,), device=actions.device)
            sigmas = self.scheduler.sigmas.to(device=actions.device)[ids].to(dtype=actions.dtype)
            noisy = self.scheduler.add_noise(clean, noise, sigmas[:, None, None]).masked_fill(~valid_mask, 0)
            timestep = self.scheduler.timesteps.to(actions.device)[ids]
            prediction = self._forward_raw(vl_embs_list, noisy, valid_mask, timestep, state, vl_attention_mask, urdf_condition)
            target = self.scheduler.training_target(clean, noise).masked_fill(~valid_mask, 0)
            error = ((prediction - target).abs() if self.loss_type == "l1" else (prediction - target).square()).masked_fill(~valid_mask, 0)
            weights = self.scheduler.training_weight(ids).to(device=actions.device, dtype=error.dtype)
            counts = valid_mask.flatten(1).sum(1).clamp_min(1)
            loss = ((error.flatten(1).sum(1) / counts) * weights).mean()
        return {"action_loss": loss, "continuous_loss": loss.detach(), "loss_mask": valid_mask, "timestep_ids": timestep.detach()}

    @torch.inference_mode()
    def predict(self, vl_embs_list, valid_mask, state=None, vl_attention_mask=None, num_inference_steps=None, urdf_condition=None, initial_sample=None):
        valid_mask = binary_bool_mask(valid_mask, device=valid_mask.device, name="continuous_action_valid_mask")
        if valid_mask.ndim != 3 or tuple(valid_mask.shape[1:]) != (self.action_horizon, self.action_dim):
            raise ValueError("continuous valid_mask has the wrong shape")
        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        if steps < 1: raise ValueError("continuous inference steps must be positive")
        compute_dtype = vl_embs_list[0].dtype
        if initial_sample is None:
            sample = torch.randn(valid_mask.shape, device=valid_mask.device, dtype=compute_dtype).masked_fill(~valid_mask, 0)
        else:
            if initial_sample.shape != valid_mask.shape or not torch.is_floating_point(initial_sample):
                raise ValueError("initial_sample must match continuous valid_mask shape and be floating point")
            sample = initial_sample.to(device=valid_mask.device, dtype=compute_dtype).masked_fill(~valid_mask, 0)
        if self.flow_matching_style == "starvla":
            dt = 1.0 / steps
            for step in range(steps):
                timestep = torch.full((valid_mask.shape[0],), int(step / float(steps) * self.num_timestep_buckets), device=valid_mask.device, dtype=torch.long)
                prediction = self._forward_raw(vl_embs_list, sample, valid_mask, timestep, state, vl_attention_mask, urdf_condition)
                sample = (sample + dt * prediction).masked_fill(~valid_mask, 0)
            return sample.clamp(-2.2, 2.2)
        scheduler = ActionScheduler()
        scheduler.set_timesteps(steps, training=False)
        sigmas = scheduler.sigmas.to(device=valid_mask.device)
        for idx, sigma in enumerate(sigmas):
            sigma_next = sigmas[idx + 1] if idx + 1 < len(sigmas) else torch.zeros_like(sigma)
            timestep = (sigma * scheduler.num_train_timesteps).expand(valid_mask.shape[0])
            prediction = self._forward_raw(vl_embs_list, sample, valid_mask, timestep, state, vl_attention_mask, urdf_condition)
            sample = scheduler.flow_step(prediction, sigma, sigma_next, sample).masked_fill(~valid_mask, 0)
        return sample.clamp(-2.2, 2.2)

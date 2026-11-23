"""Learned-code MaskGIT adapter around the pinned official starVLA DiT.

Unlike the upstream uniform ActionBinning head, inputs and outputs here are
categorical local code IDs. No bin-center loss or numeric arithmetic on IDs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
import math
import torch
from torch import nn
from torch.nn import functional as F
from robomorpher.utils.model.masking import binary_bool_mask


class DiscreteCodeHead(nn.Module):
    """Categorical MaskGIT with per-slot vocabulary and explicit padding.

    token_ids/valid_mask: [B, action_horizon, len(vocab_sizes)].
    Invalid positions return -1. MASK and PAD have separate input embeddings.
    A fixed slot position identifies its arm, latent/joint, RVQ/PQ stage or grip.
    The frozen tokenizer itself is owned by the framework, not this head.
    """

    def __init__(
        self, *, vocab_sizes: Sequence[int], action_horizon: int,
        vl_hidden_dim: int, num_layers: int, hidden_dim: int | None = None,
        num_heads: int | None = None, state_dim: int = 0, num_inference_steps: int = 8,
        num_timestep_buckets: int = 1000,
        secondary_remask: bool = False, secondary_remask_threshold: float = 0.5,
        df_timestep_semantics: bool = False, maskgit_enabled: bool = False,
        temporal_decode: bool = False, temporal_train: bool = True,
        temporal_train_mode: str = "block_ar",
        temporal_block_size: int = 1, temporal_kv_cache: bool = True,
        temporal_rvq_order: str = "coarse_to_fine",
        temporal_stage_ids: Sequence[int] | None = None,
        dropout: float | None = None,
        future_tokens: int = 32, max_seq_len: int | None = None,
        add_pos_embed: bool = True,
        diffusion_model_cfg: Mapping | None = None,
        urdf_feature_dim: int = 0, urdf_num_tokens: int = 0,
        update_urdf_tokens: bool = True,
        urdf_injection: str = "tokens", urdf_slot_map=None,
        discrete_mode: str = "categorical",
    ):
        super().__init__()
        if discrete_mode not in {"categorical", "joint_maskgit"}:
            raise ValueError("discrete_mode must be categorical or joint_maskgit")
        self.discrete_mode = str(discrete_mode)
        self.joint_maskgit = self.discrete_mode == "joint_maskgit"
        if temporal_decode or temporal_stage_ids is not None:
            raise ValueError("BD3LM and temporal RVQ decoding were removed")
        if self.joint_maskgit:
            if secondary_remask:
                raise ValueError("joint_maskgit uses DDVLA masking without secondary remask")
            if urdf_feature_dim and urdf_injection != "adaln":
                raise ValueError("joint_maskgit retains URDF AdaLN, not additional URDF prefix tokens")
        vocab_sizes = tuple(int(k) for k in vocab_sizes)
        if not vocab_sizes or min(vocab_sizes) < 2:
            raise ValueError("Every categorical slot needs at least 2 classes")
        output_dim = max(vocab_sizes)
        if action_horizon < 1 or num_layers < 1 or num_inference_steps < 1:
            raise ValueError("horizon, layers and inference steps must be positive")
        if (isinstance(num_timestep_buckets, bool) or int(num_timestep_buckets) != num_timestep_buckets or int(num_timestep_buckets) < 2):
            raise ValueError("num_timestep_buckets must be an integer >= 2")
        if state_dim < 0 or future_tokens < 0:
            raise ValueError("state_dim and future_tokens cannot be negative")
        if diffusion_model_cfg is not None and not isinstance(diffusion_model_cfg, Mapping):
            raise ValueError("diffusion_model_cfg must be a mapping")
        dit_cfg = dict(diffusion_model_cfg or {})
        self.backbone_type = "categorical_dit"
        allowed = {"num_layers", "attention_head_dim", "cross_attention_dim", "output_dim", "dropout", "final_dropout", "interleave_self_attention",
                   "norm_type", "positional_embeddings", "max_num_positional_embeddings",
                   "attention_bias", "activation_fn", "upcast_attention", "norm_elementwise_affine", "norm_eps"}
        unknown = set(dit_cfg) - allowed
        if unknown:
            raise ValueError(f"Unsupported diffusion_model_cfg settings: {sorted(unknown)}")
        head_dim = dit_cfg.pop("attention_head_dim", None)
        for name, value in (("hidden_dim", hidden_dim), ("num_heads", num_heads),
                            ("attention_head_dim", head_dim)):
            if value is not None and (isinstance(value, bool) or int(value) <= 0):
                raise ValueError(f"{name} must be positive")
        hidden_dim = int(hidden_dim or vl_hidden_dim)
        num_heads = int(num_heads or (hidden_dim // int(head_dim) if head_dim else 8))
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by positive num_heads")
        if head_dim is not None and int(head_dim) != hidden_dim // num_heads:
            raise ValueError("attention_head_dim conflicts with hidden_dim / num_heads")
        for name, value in (("num_layers", num_layers), ("cross_attention_dim", vl_hidden_dim), ("output_dim", output_dim)):
            configured = dit_cfg.pop(name, value)
            if int(configured) != int(value):
                raise ValueError(f"diffusion_model_cfg.{name} conflicts with the resolved VLM/codec/head dimensions")
        nested_dropout = dit_cfg.pop("dropout", None)
        if dropout is not None and nested_dropout is not None and float(dropout) != float(nested_dropout):
            raise ValueError("dropout conflicts with diffusion_model_cfg.dropout")
        dropout = float(dropout if dropout is not None else nested_dropout if nested_dropout is not None else 0.0)
        if not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be finite in [0, 1]")
        if not isinstance(add_pos_embed, bool):
            raise ValueError("add_pos_embed must be boolean")
        if not isinstance(update_urdf_tokens, bool):
            raise ValueError("update_urdf_tokens must be boolean")
        interleave = dit_cfg.pop("interleave_self_attention", True)
        final_dropout = dit_cfg.pop("final_dropout", bool(dropout))
        if not isinstance(interleave, bool) or not isinstance(final_dropout, bool):
            raise ValueError("interleave_self_attention and final_dropout must be boolean")
        norm_type = dit_cfg.pop("norm_type", "ada_norm")
        if norm_type not in ("ada_norm", "layer_norm"):
            raise ValueError("This DiT implements norm_type=ada_norm or layer_norm")
        positional_embeddings = dit_cfg.pop("positional_embeddings", None if self.joint_maskgit else "sinusoidal")
        if positional_embeddings not in (None, "sinusoidal"):
            raise ValueError("positional_embeddings must be sinusoidal or null")
        if self.joint_maskgit and positional_embeddings is not None:
            raise ValueError("joint_maskgit uses DF's learned action positions without block positional embeddings")
        self.train_mask_schedule = "cosine"
        self.decode_schedule = "cosine"
        self.add_pos_embed = add_pos_embed
        self.interleave_self_attention = interleave
        # Deliberately lazy: tokenizer-only jobs do not need VLA dependencies.
        from robomorpher.model.backbones.action.categorical_dit import DiT

        self.vocab_sizes = vocab_sizes
        self.num_slots = len(vocab_sizes)
        self.action_horizon = int(action_horizon)
        self.num_layers = int(num_layers)
        self.num_inference_steps = int(num_inference_steps)
        # Official DF uses a 1000-bucket integer time embedding. Keep the
        # bucket count explicit so training and denoising share one contract.
        self.num_timestep_buckets = int(num_timestep_buckets)
        if not isinstance(secondary_remask, bool):
            raise ValueError("secondary_remask must be boolean")
        if (isinstance(secondary_remask_threshold, bool)
                or not math.isfinite(float(secondary_remask_threshold))
                or not 0.0 <= float(secondary_remask_threshold) <= 1.0):
            raise ValueError("secondary_remask_threshold must be finite in [0, 1]")
        # Official DF has no secondary re-masking pass. Keep the argument only
        # for old call sites, but make the strict default deterministic.
        # Official DF has no secondary re-masking; the argument is
        # accepted only to keep old callers import-compatible, but never used.
        self.secondary_remask = False
        self.secondary_remask_threshold = float(secondary_remask_threshold)
        self.df_timestep_semantics = bool(df_timestep_semantics)
        if not isinstance(maskgit_enabled, bool):
            raise ValueError("maskgit_enabled must be boolean")
        if maskgit_enabled and not self.joint_maskgit:
            raise ValueError(
                "MaskGIT iterative decoding was removed. Discrete tokens use full-mask CE or BD3LM, then argmax."
            )
        self.maskgit_enabled = self.joint_maskgit
        if not isinstance(temporal_decode, bool) or not isinstance(temporal_train, bool):
            raise ValueError("temporal_decode and temporal_train must be boolean")
        if temporal_train_mode not in ("block_ar", "bd3lm"):
            raise ValueError("temporal_train_mode must be block_ar or bd3lm")
        if not isinstance(temporal_kv_cache, bool):
            raise ValueError("temporal_kv_cache must be boolean")
        if isinstance(temporal_block_size, bool) or int(temporal_block_size) < 1:
            raise ValueError("temporal_block_size must be a positive integer")
        if int(temporal_block_size) > int(action_horizon):
            raise ValueError("temporal_block_size cannot exceed action_horizon")
        if temporal_rvq_order not in ("coarse_to_fine",):
            raise ValueError("Only temporal_rvq_order=coarse_to_fine is supported")
        if temporal_decode:
            if temporal_stage_ids is None or len(temporal_stage_ids) != self.num_slots:
                raise ValueError("Temporal decoding requires one RVQ stage ID per token slot")
            if any(isinstance(x, bool) or int(x) < 0 for x in temporal_stage_ids):
                raise ValueError("Temporal stage IDs must be nonnegative integers")
            if max(int(x) for x in temporal_stage_ids) < 1:
                raise ValueError("Temporal RVQ decoding requires at least two code levels")
            self.register_buffer(
                "temporal_stage_ids",
                torch.as_tensor(tuple(int(x) for x in temporal_stage_ids), dtype=torch.long),
                persistent=False,
            )
        else:
            self.register_buffer("temporal_stage_ids", torch.zeros(self.num_slots, dtype=torch.long), persistent=False)
        self.temporal_decode = bool(temporal_decode)
        self.temporal_train = bool(temporal_train)
        self.temporal_train_mode = str(temporal_train_mode)
        self.temporal_kv_cache = bool(temporal_kv_cache)
        self.temporal_block_size = int(temporal_block_size)
        self.temporal_rvq_order = str(temporal_rvq_order)
        self.num_bins = max(vocab_sizes)
        self.mask_token_id = self.num_bins
        self.pad_token_id = self.num_bins + 1
        self.state_dim = int(state_dim)
        if (isinstance(urdf_feature_dim, bool) or isinstance(urdf_num_tokens, bool)
                or int(urdf_feature_dim) != urdf_feature_dim or int(urdf_num_tokens) != urdf_num_tokens
                or urdf_feature_dim < 0 or urdf_num_tokens < 0
                or bool(urdf_feature_dim) != bool(urdf_num_tokens)):
            raise ValueError("urdf_feature_dim and urdf_num_tokens must both be positive or both zero")
        self.urdf_feature_dim, self.urdf_num_tokens = int(urdf_feature_dim), int(urdf_num_tokens)
        if urdf_injection not in {"tokens", "adaln"}:
            raise ValueError("urdf_injection must be tokens or adaln")
        self.urdf_injection = urdf_injection
        if urdf_injection == "adaln":
            if not urdf_feature_dim:
                raise ValueError("URDF AdaLN requires encoded URDF features")
            if urdf_slot_map is None or len(urdf_slot_map) != self.num_slots:
                raise ValueError("URDF AdaLN requires one explicit geometry mapping per action slot")
            weights = torch.zeros(self.num_slots, self.urdf_num_tokens)
            for slot, indices in enumerate(urdf_slot_map):
                if not indices or any(not isinstance(j, int) or j < 0 or j >= self.urdf_num_tokens for j in indices):
                    raise ValueError("Invalid URDF slot mapping")
                weights[slot, indices] = 1.0
            self.register_buffer("urdf_slot_weights", weights, persistent=False)
        # The projector remains trainable in both modes. When disabled, only the
        # per-layer state update is suppressed; action tokens can still attend to
        # the same encoded URDF prefix and backpropagate through its projector.
        self.update_urdf_tokens = bool(update_urdf_tokens)
        self.hidden_dim = hidden_dim
        self.num_heads = int(num_heads)
        self.vl_hidden_dim = int(vl_hidden_dim)
        self.seq_len = self.action_horizon * self.num_slots
        self.register_buffer("slot_vocab_sizes", torch.tensor(vocab_sizes, dtype=torch.long))
        prefix_len = int(future_tokens) + int(state_dim > 0) + (self.urdf_num_tokens if urdf_injection == "tokens" else 0)
        required_seq_len = self.seq_len * (2 if temporal_train_mode == "bd3lm" else 1) + prefix_len
        if max_seq_len is not None and int(max_seq_len) < required_seq_len:
            raise ValueError(f"max_seq_len {max_seq_len} < required {required_seq_len}")
        positional_capacity = int(dit_cfg.pop("max_num_positional_embeddings", max_seq_len or required_seq_len))
        if positional_capacity < 1:
            raise ValueError("max_num_positional_embeddings must be positive")
        if positional_embeddings is not None and positional_capacity < required_seq_len:
            raise ValueError("max_num_positional_embeddings cannot hold action and prefix tokens")
        self.model = DiT(
            num_layers=self.num_layers, num_attention_heads=num_heads,
            attention_head_dim=hidden_dim // num_heads, output_dim=output_dim,
            cross_attention_dim=int(vl_hidden_dim), norm_type=norm_type,
            dropout=dropout, final_dropout=final_dropout,
            interleave_self_attention=interleave, positional_embeddings=positional_embeddings,
            max_num_positional_embeddings=positional_capacity, **dit_cfg,
        )
        self.token_embedding = nn.Embedding(self.num_bins + 2, hidden_dim, padding_idx=self.pad_token_id)
        if self.joint_maskgit:
            # DF's discrete action encoder: local code embedding concatenated
            # with the sinusoidal discrete-time code, followed by its two MLPs.
            # The categorical embedding keeps our codec-specific MASK/PAD IDs.
            from robomorpher.model.backbones.action.continuous_action_head import _SinusoidalActionTime
            self.joint_action_time = _SinusoidalActionTime(hidden_dim)
            self.joint_time_layer2 = nn.Linear(2 * hidden_dim, hidden_dim)
            self.joint_time_layer3 = nn.Linear(hidden_dim, hidden_dim)
            self.joint_decoder = nn.Sequential(
                nn.Linear(hidden_dim, 1024), nn.ReLU(), nn.Linear(1024, output_dim))
        self.position_embedding = nn.Embedding(positional_capacity, hidden_dim) if add_pos_embed else None
        self.future_tokens = nn.Parameter(torch.empty(int(future_tokens), hidden_dim))
        self.state_encoder = (
            nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            if state_dim else None
        )
        # The encoder stays owned/frozen by the codec; only this projector trains.
        self.urdf_projector = (
            nn.Linear(self.urdf_feature_dim, hidden_dim)
            if self.urdf_feature_dim and urdf_injection == "tokens" else None
        )
        if urdf_injection == "adaln":
            for block in self.model.transformer_blocks:
                block.urdf_modulation = nn.Sequential(
                    nn.SiLU(), nn.Linear(self.urdf_feature_dim, 6 * hidden_dim))
                nn.init.zeros_(block.urdf_modulation[-1].weight)
                nn.init.zeros_(block.urdf_modulation[-1].bias)
        nn.init.normal_(self.token_embedding.weight, std=.02)
        with torch.no_grad():
            self.token_embedding.weight[self.pad_token_id].zero_()
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight, std=.02)
        nn.init.normal_(self.future_tokens, std=.02)

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing = False

    def _check_layout(self, valid_mask, token_ids=None):
        if valid_mask.ndim != 3 or tuple(valid_mask.shape[1:]) != (self.action_horizon, self.num_slots):
            raise ValueError("valid_mask must have [B, horizon, num_slots] shape")
        valid_mask = binary_bool_mask(valid_mask, name="valid_mask")
        if token_ids is not None:
            if token_ids.shape != valid_mask.shape:
                raise ValueError("token_ids and valid_mask shape mismatch")
            if token_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise TypeError("Learned code IDs must be integer tensors, never normalized floats")
            k = self.slot_vocab_sizes.view(1, 1, -1)
            if ((token_ids < 0) & valid_mask).any() or ((token_ids >= k) & valid_mask).any():
                raise ValueError("A valid token ID is outside its slot vocabulary")
        return valid_mask

    @staticmethod
    def _key_mask(valid, dtype):
        return torch.zeros(valid.shape[0], 1, valid.shape[1], device=valid.device, dtype=dtype).masked_fill(
            ~valid[:, None, :], torch.finfo(dtype).min
        )

    def _temporal_attention_mask(self, full_valid, prefix_len, dtype):
        """Build block-causal action self-attention for temporal MaskGIT.

        Prefix queries only attend to valid prefix keys. An action query in
        temporal block i can attend to all earlier temporal blocks and to the
        current block's tokens whose RVQ stage is no later than its own stage.
        Thus tokens within one stage remain parallel, RVQ2 can consume RVQ1,
        and future temporal blocks are completely absent from the key set.
        The returned additive mask is [B * num_heads, Q, K], which is the
        shape accepted by diffusers' Attention processors without broadcasting
        a query-independent padding mask.
        """
        if not self.temporal_decode:
            raise RuntimeError("Temporal attention mask requested when disabled")
        if prefix_len < 0 or prefix_len > full_valid.shape[1]:
            raise ValueError("prefix_len is outside the self-attention sequence")
        batch, total = full_valid.shape
        device = full_valid.device
        base = torch.zeros((total, total), dtype=torch.bool, device=device)
        if prefix_len:
            base[:prefix_len, :prefix_len] = True
            base[prefix_len:, :prefix_len] = True

        action_len = total - prefix_len
        if action_len != self.seq_len:
            raise ValueError("Temporal action sequence length does not match the head layout")
        action_time = torch.arange(self.action_horizon, device=device).repeat_interleave(self.num_slots)
        action_stage = self.temporal_stage_ids.to(device=device).repeat(self.action_horizon)
        action_block = action_time // self.temporal_block_size
        relation = (
            (action_block[:, None] > action_block[None, :])
            | (
                (action_block[:, None] == action_block[None, :])
                & (action_stage[None, :] <= action_stage[:, None])
            )
        )
        base[prefix_len:, prefix_len:] = relation

        # Padding keys are never visible. Give invalid query rows a diagonal
        # sentinel so scaled_dot_product_attention never receives an all-masked
        # row; these rows are zeroed after every block and never produce loss.
        key_valid = full_valid.clone()
        empty = ~key_valid.any(-1)
        if bool(empty.any()):
            key_valid[empty, 0] = True
        allow = base[None] & key_valid[:, None, :]
        diagonal = torch.eye(total, dtype=torch.bool, device=device)[None]
        allow = allow | ((~full_valid)[:, :, None] & diagonal)
        additive = torch.zeros((batch, total, total), device=device, dtype=dtype)
        additive = additive.masked_fill(~allow, torch.finfo(dtype).min)
        return additive.repeat_interleave(self.num_heads, dim=0)

    def _bd3lm_position_ids(self, prefix_len, device):
        """Absolute positions for the doubled noisy and clean action copies.

        The deployed RVQ checkpoint was trained with ordinary sequential
        positions. The clean copy therefore keeps its second-half indices.
        """
        return torch.arange(prefix_len + 2 * self.seq_len, device=device)

    def _bd3lm_attention_mask(self, full_valid, prefix_len, dtype):
        """BD3LM mask for ``[prefix, noisy xt, clean x0]``.

        The noisy copy is predicted. A noisy query in block b sees the
        prefix, noisy queries in its own block, and clean blocks strictly
        before b. Clean queries see clean blocks up to their own block.
        """
        batch, total = full_valid.shape
        n = self.seq_len
        if total - prefix_len != 2 * n:
            raise ValueError("BD3LM layout must contain exactly noisy and clean action copies")
        device = full_valid.device
        noisy_start = prefix_len
        clean_start = prefix_len + n
        base = torch.zeros((total, total), dtype=torch.bool, device=device)
        if prefix_len:
            base[:prefix_len, :prefix_len] = True
            base[noisy_start:, :prefix_len] = True
        stages = self.temporal_stage_ids.to(device=device).repeat(self.action_horizon)
        # Clean copy: block-causal clean context, including current block.
        c_rel = stages[:, None] >= stages[None, :]
        base[clean_start:, clean_start:] = c_rel
        # Noisy copy: own noisy block plus clean strictly earlier blocks.
        n_rel = stages[:, None] == stages[None, :]
        base[noisy_start:clean_start, noisy_start:clean_start] = n_rel
        base[noisy_start:clean_start, clean_start:] = stages[:, None] > stages[None, :]
        key_valid = full_valid.clone()
        empty = ~key_valid.any(-1)
        if bool(empty.any()):
            key_valid[empty, 0] = True
        allow = base[None] & key_valid[:, None, :]
        diagonal = torch.eye(total, dtype=torch.bool, device=device)[None]
        allow = allow | ((~full_valid)[:, :, None] & diagonal)
        return torch.zeros((batch, total, total), device=device, dtype=dtype).masked_fill(
            ~allow, torch.finfo(dtype).min
        ).repeat_interleave(self.num_heads, dim=0)

    def _block_ar_attention_mask(self, full_valid, prefix_len, dtype):
        """Strict same-block RVQ stage-causal attention for one training pass."""
        batch, total = full_valid.shape
        if prefix_len < 0 or prefix_len > total:
            raise ValueError("prefix_len is outside the self-attention sequence")
        if total - prefix_len != self.seq_len:
            raise ValueError("Temporal action sequence length does not match the head layout")
        device = full_valid.device
        base = torch.zeros((total, total), dtype=torch.bool, device=device)
        if prefix_len:
            base[:prefix_len, :prefix_len] = True
            base[prefix_len:, :prefix_len] = True
        time = torch.arange(self.action_horizon, device=device).repeat_interleave(self.num_slots)
        stage = self.temporal_stage_ids.to(device=device).repeat(self.action_horizon)
        block = time // self.temporal_block_size
        relation = (block[:, None] > block[None, :]) | (
            (block[:, None] == block[None, :])
            & (stage[:, None] > stage[None, :])
        )
        base[prefix_len:, prefix_len:] = relation
        key_valid = full_valid.clone()
        empty = ~key_valid.any(-1)
        if bool(empty.any()):
            key_valid[empty, 0] = True
        allow = base[None] & key_valid[:, None, :]
        diagonal = torch.eye(total, dtype=torch.bool, device=device)[None]
        allow = allow | ((~full_valid)[:, :, None] & diagonal)
        additive = torch.zeros((batch, total, total), device=device, dtype=dtype)
        return additive.masked_fill(~allow, torch.finfo(dtype).min).repeat_interleave(self.num_heads, dim=0)

    def _logits(self, vl_embs_list, token_ids, valid_mask, state=None, vl_attention_mask=None, *, kv_cache=None, global_timestep=None, discrete_timestep=None, continuous_timestep=None, self_attention_mask=None, block_ar_training=False, clean_token_ids=None):
        """Use the official blocks; explicitly supply masks to cross AND self attention.

        The upstream DiT.forward passes no self-attention padding mask. Calling its
        same blocks here preserves the upstream block computation and prevents invalid action
        slots from leaking into valid slots.
        """
        if self.joint_maskgit:
            if clean_token_ids is not None or kv_cache is not None or block_ar_training or self_attention_mask is not None:
                raise ValueError("joint_maskgit forbids clean copies, action caches and stage attention overrides")
            if not getattr(self.model, "uses_df_token_routing", False):
                raise RuntimeError("joint_maskgit requires the DF per-token hybrid owner")
            # One source of truth for both action-input time and AdaLN time.
            # Never accept an externally sampled time unrelated to the input.
            discrete_timestep, global_timestep = self._joint_time_from_input(token_ids, valid_mask)
            continuous_timestep = torch.zeros_like(discrete_timestep)
        if kv_cache is not None and kv_cache.get("temporal"):
            logits, stage_kv = self._logits_temporal_cached(
                token_ids, valid_mask, kv_cache,
                kv_cache["block_index"], kv_cache["stage"],
            )
            kv_cache["last_stage_kv"] = stage_kv
            cache_data = kv_cache["data"]
            start = int(kv_cache["block_index"]) * self.temporal_block_size
            end = min(self.action_horizon, start + self.temporal_block_size)
            stage_slots = self.temporal_stage_ids.to(token_ids.device) == int(kv_cache["stage"])
            positions = torch.arange(start, end, device=token_ids.device)[:, None] * self.num_slots + torch.arange(self.num_slots, device=token_ids.device)[None, :]
            positions = positions[stage_slots.expand(positions.shape[0], -1)].reshape(-1)
            kv_cache["last_current_valid"] = valid_mask.reshape(valid_mask.shape[0], -1).index_select(1, positions)
            return logits
        batch = token_ids.shape[0]
        device = token_ids.device
        flat_valid = valid_mask.reshape(batch, -1)
        safe = torch.where(flat_valid, token_ids.reshape(batch, -1), self.pad_token_id)
        hidden = self.token_embedding(safe)
        if self.joint_maskgit:
            time = self.joint_action_time(discrete_timestep).to(hidden.dtype)
            encoded = self.joint_time_layer2(torch.cat(
                (hidden, time[:, None].expand(-1, self.seq_len, -1)), dim=-1))
            hidden = self.joint_time_layer3(F.silu(encoded))
        bd3lm = clean_token_ids is not None
        clean_flat_valid = None
        clean_hidden = None
        if bd3lm:
            if clean_token_ids.shape != token_ids.shape:
                raise ValueError("clean_token_ids must have the same [B,H,S] shape as noisy token_ids")
            clean_flat_valid = flat_valid
            clean_safe = torch.where(clean_flat_valid, clean_token_ids.reshape(batch, -1), self.pad_token_id)
            clean_hidden = self.token_embedding(clean_safe)
        position_embedding = (self.model.shared_position_embedding if self.joint_maskgit
                              else self.position_embedding)
        if position_embedding is not None:
            position_offset = self.model.continuous_horizon if self.joint_maskgit else 0
            noisy_pos = position_embedding(
                torch.arange(position_offset, position_offset + self.seq_len, device=device)
            )[None]
            hidden = hidden + noisy_pos
            if bd3lm:
                clean_pos = self.position_embedding(
                    torch.arange(self.seq_len, 2 * self.seq_len, device=device)
                )[None]
                clean_hidden = clean_hidden + clean_pos
        hidden = torch.where(flat_valid[..., None], hidden, torch.zeros_like(hidden))
        if bd3lm:
            clean_hidden = torch.where(clean_flat_valid[..., None], clean_hidden, torch.zeros_like(clean_hidden))
        pieces, masks = [], []
        urdf_condition = None
        if self.urdf_feature_dim:
            if not isinstance(state, Mapping) or set(state) != {"urdf_features", "urdf_valid_mask", "state"}:
                raise ValueError("Frozen URDF conditioning requires features, valid mask and optional state")
            features = state["urdf_features"]
            uv = binary_bool_mask(state["urdf_valid_mask"], device=device, name="urdf_valid_mask")
            if (features.shape != (batch, self.urdf_num_tokens, self.urdf_feature_dim)
                    or uv.shape != (batch, self.urdf_num_tokens)):
                raise ValueError("URDF feature/mask shape does not match the codec contract")
            features = torch.where(uv[..., None], features, torch.zeros_like(features))
            if not bool(torch.isfinite(features).all()):
                raise ValueError("Nonfinite valid URDF feature")
            if self.urdf_injection == "adaln":
                weights = self.urdf_slot_weights[None] * uv[:, None].to(features.dtype)
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
                per_slot = torch.bmm(weights, features)
                urdf_action = per_slot[:, None].expand(
                    -1, self.action_horizon, -1, -1).reshape(batch, self.seq_len, self.urdf_feature_dim)
                urdf_condition = torch.cat((urdf_action, urdf_action), dim=1) if bd3lm else urdf_action
                urdf_global = features.sum(1) / uv.sum(1, keepdim=True).clamp_min(1)
            else:
                projected = self.urdf_projector(features.to(hidden.dtype))
                pieces.append(torch.where(uv[..., None], projected, torch.zeros_like(projected)))
                masks.append(uv)
            state = state["state"]
        elif isinstance(state, Mapping):
            raise ValueError("URDF token features supplied to a head without a URDF projector")
        if self.state_encoder is not None:
            if state is None:
                raise ValueError("state is required when state_dim > 0")
            if state.ndim == 3 and state.shape[1] == 1:
                state = state[:, 0]
            if tuple(state.shape) != (batch, self.state_dim) or not torch.isfinite(state).all():
                raise ValueError("state must be finite [B, state_dim]")
            pieces.append(self.state_encoder(state.to(hidden.dtype))[:, None])
            masks.append(torch.ones(batch, 1, dtype=torch.bool, device=device))
        elif state is not None:
            raise ValueError("state supplied to a head with state_dim=0")
        # FSDP may expose this shared parameter as a flattened 1-D view.
        # Restore [num_future_tokens, hidden_dim] before sequence assembly so
        # the discrete branch has the same logical layout under 1 or N GPUs.
        future_tokens = self.model.shared_future_tokens if self.joint_maskgit else self.future_tokens
        if future_tokens.ndim != 2 or future_tokens.shape[-1] != self.hidden_dim:
            if future_tokens.numel() % self.hidden_dim:
                raise RuntimeError("FSDP-flattened future_tokens is not divisible by hidden_dim")
            future_tokens = future_tokens.reshape(-1, self.hidden_dim)
        if future_tokens.shape[0]:
            pieces.append(future_tokens[None].expand(batch, -1, -1))
            masks.append(torch.ones(batch, future_tokens.shape[0], dtype=torch.bool, device=device))
        prefix_len = sum(p.shape[1] for p in pieces)
        pieces.append(hidden)
        masks.append(flat_valid)
        if bd3lm:
            pieces.append(clean_hidden)
            masks.append(clean_flat_valid)
        hidden = torch.cat(pieces, 1)
        full_valid = torch.cat(masks, 1)
        position_ids = self._bd3lm_position_ids(prefix_len, device) if bd3lm else None
        if urdf_condition is not None:
            urdf_condition = torch.cat((
                urdf_global[:, None].expand(-1, prefix_len, -1), urdf_condition), dim=1)
            urdf_condition = urdf_condition.to(hidden.dtype).masked_fill(~full_valid[..., None], 0)
        frozen_urdf = (
            hidden[:, :self.urdf_num_tokens].clone()
            if self.urdf_projector is not None and not self.update_urdf_tokens else None
        )
        self_key_valid = full_valid.clone()
        empty = ~self_key_valid.any(-1)
        if bool(empty.any()):
            self_key_valid[empty, 0] = True
        # Temporal block/stage causality is an inference-time rollout layout.
        # When temporal_train=false (strict DF training), train the discrete
        # branch on the ordinary full token context; retaining the block mask
        # here would silently train a different causal model than DF.
        use_temporal_layout = (not self.joint_maskgit and self.temporal_decode
                               and (self.temporal_train or not self.training))
        self_mask = (
            self_attention_mask.to(dtype=hidden.dtype) if self_attention_mask is not None else
            (self._bd3lm_attention_mask(full_valid, prefix_len, hidden.dtype) if bd3lm else
             (self._block_ar_attention_mask(full_valid, prefix_len, hidden.dtype)
              if (block_ar_training and self.temporal_decode)
              else (self._temporal_attention_mask(full_valid, prefix_len, hidden.dtype)
                    if use_temporal_layout else self._key_mask(self_key_valid, hidden.dtype))))
        )
        layers = list(vl_embs_list) if isinstance(vl_embs_list, (list, tuple)) else [vl_embs_list] * self.num_layers
        if len(layers) < self.num_layers:
            raise ValueError("Not enough layer-wise VLM hidden states")
        layers = layers[-self.num_layers:]
        for i, h in enumerate(layers):
            is_self = self.interleave_self_attention and i % 2 == 1
            if h is None:
                if not is_self:
                    raise ValueError(f"Missing VLM hidden state for cross-attention layer {i}")
                continue
            if h.ndim != 3 or h.shape[0] != batch or h.shape[-1] != self.vl_hidden_dim:
                raise ValueError("VLM hidden states must have [B, sequence, hidden] shape matching cross_attention_dim")
        valid_vl = None
        cross_mask = None
        if vl_attention_mask is not None:
            valid_vl = binary_bool_mask(vl_attention_mask, device=device, name="vl_attention_mask")
            if any(tuple(h.shape[:2]) != tuple(valid_vl.shape) for h in layers if h is not None) or not valid_vl.any(-1).all():
                raise ValueError("VLM mask shape mismatch or entirely padded prompt")
            layers = [torch.where(valid_vl[..., None], h, torch.zeros_like(h)) if h is not None else None for h in layers]
            cross_mask = self._key_mask(valid_vl, hidden.dtype)

        if getattr(self.model, "is_hybrid", False):
            if global_timestep is None:
                global_timestep = torch.zeros(batch, dtype=torch.long, device=device)
            else:
                global_timestep = global_timestep.to(device=device, dtype=torch.long)
            if discrete_timestep is None:
                discrete_timestep = global_timestep
            else:
                discrete_timestep = discrete_timestep.to(device=device, dtype=torch.long)
            if continuous_timestep is None:
                continuous_timestep = discrete_timestep
            else:
                continuous_timestep = continuous_timestep.to(device=device, dtype=torch.long)
            hidden = self.model.forward_branch(
                "discrete", hidden, layers, full_valid=full_valid,
                self_attention_mask=self_mask, cross_attention_mask=cross_mask,
                global_timestep=global_timestep, discrete_timestep=discrete_timestep,
                continuous_timestep=continuous_timestep,
                urdf_condition=urdf_condition, urdf_mask=full_valid,
                frozen_prefix=frozen_urdf,
                frozen_prefix_len=(self.urdf_num_tokens if frozen_urdf is not None else 0),
                position_ids=position_ids,
                **({"prefix_len": prefix_len} if self.joint_maskgit else {}),
            )
            action_hidden = hidden[:, prefix_len:prefix_len + self.seq_len] if bd3lm else hidden[:, prefix_len:]
            logits = self.joint_decoder(action_hidden) if self.joint_maskgit else self.model.proj_out_2(action_hidden)
            logits = logits.reshape(batch, self.action_horizon, self.num_slots, -1)
            legal = torch.arange(self.num_bins, device=device)[None, None, None, :] < self.slot_vocab_sizes[None, None, :, None]
            return logits.masked_fill(~legal, float("-inf"))

        temb = self.model.timestep_encoder(torch.zeros(batch, dtype=torch.long, device=device))
        for i, block in enumerate(self.model.transformer_blocks):
            is_self = self.interleave_self_attention and i % 2 == 1
            block_forward = partial(
                block, encoder_hidden_states=None if is_self else layers[i].to(hidden.dtype),
                encoder_attention_mask=self_mask if is_self else cross_mask, temb=temb,
                position_ids=position_ids,
                **({"urdf_condition": urdf_condition, "urdf_mask": full_valid}
                   if urdf_condition is not None else {}),
            )
            # Static VLM K/V are projected once per prediction call. The
            # training path remains the original Diffusers implementation.
            if kv_cache is not None and not self.training:
                if is_self:
                    hidden, _ = block.forward_cached(
                        hidden, attention_mask=self_mask, temb=temb,
                        position_ids=position_ids,
                        **({"urdf_condition": urdf_condition, "urdf_mask": full_valid}
                           if urdf_condition is not None else {}),
                    )
                else:
                    if kv_cache.get("cross") is None:
                        kv_cache["cross"] = [None] * self.num_layers
                    if kv_cache["cross"][i] is None:
                        kv_cache["cross"][i] = block.cache_encoder_kv(layers[i].to(hidden.dtype))
                    hidden, _ = block.forward_cached(
                        hidden, encoder_key_value=kv_cache["cross"][i],
                        encoder_attention_mask=cross_mask, temb=temb,
                        position_ids=position_ids,
                        **({"urdf_condition": urdf_condition, "urdf_mask": full_valid}
                           if urdf_condition is not None else {}),
                    )
            elif self.training and self.model.gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                hidden = checkpoint(block_forward, hidden, use_reentrant=False)
            else:
                hidden = block_forward(hidden)
            hidden = torch.where(full_valid[..., None], hidden, torch.zeros_like(hidden))
            if frozen_urdf is not None:
                # Keep the encoded URDF prefix identical across DiT layers.
                # Re-concatenation avoids an in-place autograd mutation.
                hidden = torch.cat(
                    (frozen_urdf, hidden[:, self.urdf_num_tokens:]), dim=1
                )
        shift, scale = self.model.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden = self.model.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
        logits = self.model.proj_out_2(hidden[:, prefix_len:prefix_len + self.seq_len] if bd3lm else hidden[:, prefix_len:])
        logits = logits.reshape(batch, self.action_horizon, self.num_slots, -1)
        legal = torch.arange(self.num_bins, device=device)[None, None, None, :] < self.slot_vocab_sizes[None, None, :, None]
        return logits.masked_fill(~legal, float("-inf"))


    def _prepare_temporal_cache_inputs(
        self, vl_embs_list, token_ids, valid_mask, state=None,
        vl_attention_mask=None,
    ):
        """Build the prefix/action tensors used by incremental temporal decoding."""
        batch, horizon, slots = token_ids.shape
        device = token_ids.device
        flat_valid = valid_mask.reshape(batch, -1)
        safe = torch.where(flat_valid, token_ids.reshape(batch, -1), self.pad_token_id)
        action_hidden = self.token_embedding(safe)
        if self.position_embedding is not None:
            action_hidden = action_hidden + self.position_embedding(
                torch.arange(self.seq_len, device=device)
            )[None]
        action_hidden = torch.where(flat_valid[..., None], action_hidden, torch.zeros_like(action_hidden))
        pieces, masks = [], []
        urdf_condition = None
        frozen_urdf = None
        if self.urdf_feature_dim:
            if not isinstance(state, Mapping) or set(state) != {"urdf_features", "urdf_valid_mask", "state"}:
                raise ValueError("Frozen URDF conditioning requires features, valid mask and optional state")
            features = state["urdf_features"]
            uv = binary_bool_mask(state["urdf_valid_mask"], device=device, name="urdf_valid_mask")
            if features.shape != (batch, self.urdf_num_tokens, self.urdf_feature_dim) or uv.shape != (batch, self.urdf_num_tokens):
                raise ValueError("URDF feature/mask shape does not match the codec contract")
            features = torch.where(uv[..., None], features, torch.zeros_like(features))
            if not bool(torch.isfinite(features).all()):
                raise ValueError("Nonfinite valid URDF feature")
            if self.urdf_injection == "adaln":
                weights = self.urdf_slot_weights[None] * uv[:, None].to(features.dtype)
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
                per_slot = torch.bmm(weights, features)
                urdf_condition = per_slot[:, None].expand(
                    -1, self.action_horizon, -1, -1
                ).reshape(batch, self.seq_len, self.urdf_feature_dim)
                urdf_global = features.sum(1) / uv.sum(1, keepdim=True).clamp_min(1)
            else:
                projected = self.urdf_projector(features.to(action_hidden.dtype))
                pieces.append(torch.where(uv[..., None], projected, torch.zeros_like(projected)))
                masks.append(uv)
            state = state["state"]
        elif isinstance(state, Mapping):
            raise ValueError("URDF token features supplied to a head without a URDF projector")
        if self.state_encoder is not None:
            if state is None:
                raise ValueError("state is required when state_dim > 0")
            if state.ndim == 3 and state.shape[1] == 1:
                state = state[:, 0]
            if tuple(state.shape) != (batch, self.state_dim) or not torch.isfinite(state).all():
                raise ValueError("state must be finite [B, state_dim]")
            pieces.append(self.state_encoder(state.to(action_hidden.dtype))[:, None])
            masks.append(torch.ones(batch, 1, dtype=torch.bool, device=device))
        elif state is not None:
            raise ValueError("state supplied to a head with state_dim=0")
        # FSDP may expose this shared parameter as a flattened 1-D view.
        # Restore [num_future_tokens, hidden_dim] before sequence assembly so
        # the discrete branch has the same logical layout under 1 or N GPUs.
        future_tokens = self.future_tokens
        if future_tokens.ndim != 2 or future_tokens.shape[-1] != self.hidden_dim:
            if future_tokens.numel() % self.hidden_dim:
                raise RuntimeError("FSDP-flattened future_tokens is not divisible by hidden_dim")
            future_tokens = future_tokens.reshape(-1, self.hidden_dim)
        if future_tokens.shape[0]:
            pieces.append(future_tokens[None].expand(batch, -1, -1))
            masks.append(torch.ones(batch, future_tokens.shape[0], dtype=torch.bool, device=device))
        prefix_len = sum(p.shape[1] for p in pieces)
        pieces.append(action_hidden)
        masks.append(flat_valid)
        hidden = torch.cat(pieces, 1)
        full_valid = torch.cat(masks, 1)
        if urdf_condition is not None:
            urdf_condition = torch.cat((
                urdf_global[:, None].expand(-1, prefix_len, -1), urdf_condition
            ), dim=1)
            urdf_condition = urdf_condition.to(hidden.dtype).masked_fill(~full_valid[..., None], 0)
        frozen_urdf = (
            hidden[:, :self.urdf_num_tokens].clone()
            if self.urdf_projector is not None and not self.update_urdf_tokens else None
        )
        layers = list(vl_embs_list) if isinstance(vl_embs_list, (list, tuple)) else [vl_embs_list] * self.num_layers
        if len(layers) < self.num_layers:
            raise ValueError("Not enough layer-wise VLM hidden states")
        layers = layers[-self.num_layers:]
        for i, h in enumerate(layers):
            is_self = self.interleave_self_attention and i % 2 == 1
            if h is None and not is_self:
                raise ValueError(f"Missing VLM hidden state for cross-attention layer {i}")
            if h is not None and (h.ndim != 3 or h.shape[0] != batch or h.shape[-1] != self.vl_hidden_dim):
                raise ValueError("VLM hidden states must have [B, sequence, hidden] shape matching cross_attention_dim")
        valid_vl = None
        cross_mask = None
        if vl_attention_mask is not None:
            valid_vl = binary_bool_mask(vl_attention_mask, device=device, name="vl_attention_mask")
            if any(tuple(h.shape[:2]) != tuple(valid_vl.shape) for h in layers if h is not None) or not valid_vl.any(-1).all():
                raise ValueError("VLM mask shape mismatch or entirely padded prompt")
            layers = [torch.where(valid_vl[..., None], h, torch.zeros_like(h)) if h is not None else None for h in layers]
            cross_mask = self._key_mask(valid_vl, hidden.dtype)
        temb = self.model.timestep_encoder(torch.zeros(batch, dtype=torch.long, device=device))
        return {
            "batch": batch, "prefix_len": prefix_len, "prefix_hidden": hidden[:, :prefix_len],
            "prefix_valid": full_valid[:, :prefix_len], "full_valid": full_valid,
            "urdf_condition": urdf_condition, "frozen_urdf": frozen_urdf,
            "layers": layers, "cross_mask": cross_mask, "valid_vl": valid_vl,
            "temb": temb, "device": device,
        }

    @torch.no_grad()
    def _init_temporal_kv_cache(
        self, cache, vl_embs_list, token_ids, valid_mask, state=None,
        vl_attention_mask=None,
    ):
        """Cache prefix self K/V and static VLM cross K/V once per rollout."""
        data = self._prepare_temporal_cache_inputs(
            vl_embs_list, token_ids, valid_mask, state, vl_attention_mask
        )
        cache["data"] = data
        cache["cross"] = [None] * self.num_layers
        cache["self"] = [None] * self.num_layers
        prefix = data["prefix_hidden"]
        prefix_valid = data["prefix_valid"]
        prefix_len = data["prefix_len"]
        if prefix_len:
            for i, block in enumerate(self.model.transformer_blocks):
                is_self = self.interleave_self_attention and i % 2 == 1
                cond = (
                    data["urdf_condition"][:, :prefix_len]
                    if data["urdf_condition"] is not None else None
                )
                if is_self:
                    key_valid = prefix_valid.clone()
                    empty = ~key_valid.any(-1)
                    if bool(empty.any()):
                        key_valid[empty, 0] = True
                    prefix, new_kv = block.forward_cached(
                        prefix, attention_mask=self._key_mask(key_valid, prefix.dtype),
                        position_ids=torch.arange(prefix_len, device=prefix.device),
                        temb=data["temb"], urdf_condition=cond, urdf_mask=prefix_valid,
                    )
                    cache["self"][i] = (new_kv[0], new_kv[1], prefix_valid.clone())
                else:
                    if cache["cross"][i] is None:
                        cache["cross"][i] = block.cache_encoder_kv(data["layers"][i].to(prefix.dtype))
                    prefix, _ = block.forward_cached(
                        prefix, encoder_key_value=cache["cross"][i],
                        encoder_attention_mask=data["cross_mask"],
                        position_ids=torch.arange(prefix_len, device=prefix.device),
                        temb=data["temb"], urdf_condition=cond, urdf_mask=prefix_valid,
                    )
                prefix = torch.where(prefix_valid[..., None], prefix, torch.zeros_like(prefix))
                if data["frozen_urdf"] is not None:
                    frozen = data["frozen_urdf"]
                    prefix = torch.cat((frozen, prefix[:, self.urdf_num_tokens:]), dim=1)
            data["prefix_hidden"] = prefix
        # A missing prefix is valid; action stage one then starts with an empty
        # past cache and its own key/value tensors.
        cache["initialized"] = True

    @torch.no_grad()
    def _logits_temporal_cached(self, token_ids, valid_mask, cache, block_index, stage):
        data = cache["data"]
        batch = data["batch"]
        start = int(block_index) * self.temporal_block_size
        end = min(self.action_horizon, start + self.temporal_block_size)
        slots = torch.arange(self.num_slots, device=token_ids.device)
        stage_slots = self.temporal_stage_ids.to(token_ids.device) == int(stage)
        local_t = torch.arange(start, end, device=token_ids.device)
        positions = local_t[:, None] * self.num_slots + slots[None, :]
        positions = positions[stage_slots.expand(positions.shape[0], -1)].reshape(-1)
        current_ids = token_ids.reshape(batch, -1).index_select(1, positions)
        current_valid = valid_mask.reshape(batch, -1).index_select(1, positions)
        current = self.token_embedding(torch.where(current_valid, current_ids, self.pad_token_id))
        if self.position_embedding is not None:
            absolute = data["prefix_len"] + positions
            current = current + self.position_embedding(positions)[None]
        current = torch.where(current_valid[..., None], current, torch.zeros_like(current))
        cur_cond = (
            data["urdf_condition"].index_select(1, data["prefix_len"] + positions)
            if data["urdf_condition"] is not None else None
        )
        new_stage_kv = [None] * self.num_layers
        past_valid_default = torch.zeros(batch, 0, dtype=torch.bool, device=token_ids.device)
        for i, block in enumerate(self.model.transformer_blocks):
            is_self = self.interleave_self_attention and i % 2 == 1
            if is_self:
                past = cache["self"][i]
                if past is None:
                    past_kv, past_valid = None, past_valid_default
                else:
                    past_kv, past_valid = (past[0], past[1]), past[2]
                key_valid = torch.cat((past_valid, current_valid), dim=1)
                empty = ~key_valid.any(-1)
                if bool(empty.any()):
                    key_valid[empty, 0] = True
                current, new_kv = block.forward_cached(
                    current, past_key_value=past_kv,
                    attention_mask=self._key_mask(key_valid, current.dtype),
                    position_ids=data["prefix_len"] + positions,
                    temb=data["temb"], urdf_condition=cur_cond, urdf_mask=current_valid,
                )
                new_stage_kv[i] = new_kv
            else:
                if cache["cross"][i] is None:
                    cache["cross"][i] = block.cache_encoder_kv(data["layers"][i].to(current.dtype))
                current, _ = block.forward_cached(
                    current, encoder_key_value=cache["cross"][i],
                    encoder_attention_mask=data["cross_mask"],
                    position_ids=data["prefix_len"] + positions,
                    temb=data["temb"], urdf_condition=cur_cond, urdf_mask=current_valid,
                )
            current = torch.where(current_valid[..., None], current, torch.zeros_like(current))
        shift, scale = self.model.proj_out_1(F.silu(data["temb"])).chunk(2, dim=1)
        current = self.model.norm_out(current) * (1 + scale[:, None]) + shift[:, None]
        stage_logits = self.model.proj_out_2(current)
        logits = torch.full(
            (batch, self.action_horizon, self.num_slots, self.num_bins),
            float("-inf"), device=token_ids.device, dtype=stage_logits.dtype,
        )
        logits.reshape(batch, -1, logits.shape[-1]).index_copy_(1, positions, stage_logits)
        legal = torch.arange(self.num_bins, device=token_ids.device)[None, None, None, :] < self.slot_vocab_sizes[None, None, :, None]
        return logits.masked_fill(~legal, float("-inf")), new_stage_kv

    @torch.no_grad()
    def _commit_temporal_stage(self, cache):
        stage_kv = cache.pop("last_stage_kv", None)
        if stage_kv is None:
            raise RuntimeError("Temporal KV cache has no completed stage to commit")
        data = cache["data"]
        for i, new_kv in enumerate(stage_kv):
            if new_kv is None:
                continue
            current_valid = cache["last_current_valid"]
            old = cache["self"][i]
            if old is None:
                cache["self"][i] = (new_kv[0], new_kv[1], current_valid.clone())
            else:
                cache["self"][i] = (
                    torch.cat((old[0], new_kv[0]), dim=2),
                    torch.cat((old[1], new_kv[1]), dim=2),
                    torch.cat((old[2], current_valid), dim=1),
                )

    def _temporal_stage_masks(self, valid_mask):
        """Return coarse (RVQ1) and refinement (RVQ2+) slot masks."""
        stage = self.temporal_stage_ids.to(device=valid_mask.device)
        coarse = (stage == 0)[None, None, :]
        refine = (stage >= 1)[None, None, :]
        return valid_mask & coarse, valid_mask & refine

    def apply_stagewise_mask(self, token_ids, valid_mask, *, stage_override):
        """Build exact two-stage no-MaskGIT RVQ training views."""
        valid_mask = self._check_layout(valid_mask, token_ids)
        if not self.temporal_decode:
            raise RuntimeError("stagewise RVQ training requires temporal_decode")
        stage = int(stage_override)
        if stage not in (0, 1):
            raise ValueError("stage_override must be 0 (RVQ1+G) or 1 (RVQ2+)")
        coarse, refine = self._temporal_stage_masks(valid_mask)
        masked = torch.where(
            valid_mask,
            torch.full_like(token_ids, self.mask_token_id, dtype=torch.long),
            torch.full_like(token_ids, self.pad_token_id, dtype=torch.long),
        )
        if stage == 1:
            # Coarse IDs may be GT or detached model predictions supplied by caller.
            masked = torch.where(coarse, token_ids.long(), masked)
            loss_mask = refine.clone()
        else:
            loss_mask = coarse.clone()
        # Fixed zero timestep matches the two fixed no-MaskGIT inference stages.
        timestep = torch.zeros(token_ids.shape[0], device=token_ids.device, dtype=torch.float32)
        return masked, loss_mask, timestep

    def apply_block_ar_mask(self, token_ids, valid_mask, *, phase_timestep=None):
        """Keep coarse GT context, mask refinement, and avoid random mask ratios."""
        valid_mask = self._check_layout(valid_mask, token_ids)
        if not self.temporal_decode:
            raise RuntimeError("block-AR input requires temporal_decode")
        _, refine = self._temporal_stage_masks(valid_mask)
        masked = torch.where(valid_mask, token_ids.long(), self.pad_token_id)
        masked = torch.where(refine, torch.full_like(masked, self.mask_token_id), masked)
        if phase_timestep is None:
            timestep = torch.zeros(token_ids.shape[0], device=token_ids.device, dtype=torch.float32)
        else:
            timestep = phase_timestep.to(device=token_ids.device, dtype=torch.float32)
            if timestep.shape != (token_ids.shape[0],) or not bool(torch.isfinite(timestep).all()):
                raise ValueError("phase_timestep must have shape [B] and be finite")
            if bool((timestep < 0).any()) or bool((timestep >= 0.5).any()):
                raise ValueError("phase_timestep must be in [0, 0.5)")
        return masked, valid_mask.clone(), timestep

    def apply_mask(self, token_ids, valid_mask, generator=None, *, return_timestep=False,
                   timestep_override=None, mask_ratio_override=None):
        """Mask every valid token. Discrete CE is computed on that full mask."""
        del generator, mask_ratio_override
        valid_mask = self._check_layout(valid_mask, token_ids)
        batch = token_ids.shape[0]
        masked = torch.where(valid_mask, torch.full_like(token_ids.long(), self.mask_token_id), self.pad_token_id)
        loss_mask = valid_mask.clone()
        if timestep_override is None:
            t = torch.zeros(batch, device=token_ids.device, dtype=torch.float32)
        else:
            t = timestep_override.to(device=token_ids.device, dtype=torch.float32)
            if t.shape != (batch,) or not bool(torch.isfinite(t).all()) or bool((t < 0).any()) or bool((t > 1).any()):
                raise ValueError("timestep_override must be finite and in [0, 1] with shape [B]")
        result = (masked, loss_mask)
        return result + (t,) if return_timestep else result

    def _joint_valid_counts(self, valid_mask):
        valid_mask = self._check_layout(valid_mask)
        counts = valid_mask.flatten(1).sum(1)
        if not bool((counts > 0).all()):
            raise ValueError("joint_maskgit requires at least one valid action token in every row")
        return valid_mask, counts

    def _joint_time_from_input(self, token_ids, valid_mask):
        """DF local/global buckets from the *current* visible-token fraction.

        Cosine's scheduler coordinate is deliberately not an embedding time.
        FP32 division keeps these coordinates independent of AMP precision.
        """
        valid_mask, counts = self._joint_valid_counts(valid_mask)
        if token_ids.shape != valid_mask.shape:
            raise ValueError("joint_maskgit token_ids/valid_mask shape mismatch")
        is_mask = token_ids == self.mask_token_id
        known = valid_mask & ~is_mask
        self._check_layout(known, token_ids)
        masked_count = (valid_mask & is_mask).flatten(1).sum(1)
        d = 1.0 - masked_count.float() / counts.float()
        discrete = (d * self.num_timestep_buckets).long().clamp(0, self.num_timestep_buckets - 1)
        global_time = (0.5 * d * self.num_timestep_buckets).long().clamp(0, self.num_timestep_buckets - 1)
        return discrete, global_time

    def apply_joint_mask(self, token_ids, valid_mask, generator=None):
        """DDVLA uniform-u cosine masking with an exact valid count per row."""
        valid_mask = self._check_layout(valid_mask, token_ids)
        valid_mask, counts = self._joint_valid_counts(valid_mask)
        batch = token_ids.shape[0]
        u = torch.rand(batch, device=token_ids.device, dtype=torch.float32, generator=generator)
        ratio = torch.cos(0.5 * math.pi * u).clamp(1e-6, 1.0)
        n_mask = torch.minimum((counts.float() * ratio).round().long().clamp_min(1), counts)
        scores = torch.rand(valid_mask.shape, device=token_ids.device,
                            dtype=torch.float32, generator=generator).flatten(1)
        scores = scores.masked_fill(~valid_mask.flatten(1), float("inf"))
        ranks = scores.argsort(dim=1, stable=True).argsort(dim=1, stable=True)
        loss_mask = (ranks < n_mask[:, None]).reshape_as(valid_mask) & valid_mask
        corrupted = torch.where(valid_mask, token_ids.long(), self.pad_token_id)
        corrupted = corrupted.masked_fill(loss_mask, self.mask_token_id)
        return corrupted, loss_mask

    def forward_joint_maskgit(self, vl_embs_list, token_ids, valid_mask, state=None,
                              vl_attention_mask=None, generator=None):
        corrupted, loss_mask = self.apply_joint_mask(token_ids, valid_mask, generator)
        discrete_time, global_time = self._joint_time_from_input(corrupted, valid_mask)
        logits = self._logits(
            vl_embs_list, corrupted, valid_mask, state, vl_attention_mask,
            discrete_timestep=discrete_time, global_timestep=global_time,
            continuous_timestep=torch.zeros_like(discrete_time))
        # DDVLA's valid-target token mean, over the complete local batch/repeats.
        # No per-row, per-arm, or per-codebook reweighting is applied.
        loss = F.cross_entropy(logits[loss_mask].float(), token_ids[loss_mask].long(), reduction="mean")
        return {"action_loss": loss, "logits": logits, "loss_mask": loss_mask,
                "discrete_timestep": discrete_time, "global_timestep": global_time}

    @torch.no_grad()
    def _predict_joint_maskgit(
        self, vl_embs_list, valid_mask, state=None, vl_attention_mask=None, *,
        initial_ids=None, fixed_mask=None, decode_temperature=1.0,
        choice_temperature=1.0, num_inference_steps=None, generator=None,
        return_history=False,
    ):
        """DDVLA categorical proposals + Gumbel-ranked committed-token locking."""
        valid_mask, valid_counts = self._joint_valid_counts(valid_mask)
        steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        if isinstance(steps, bool) or int(steps) != steps or int(steps) < 1:
            raise ValueError("joint_maskgit inference steps must be a positive integer")
        steps = int(steps)
        fixed = torch.zeros_like(valid_mask) if fixed_mask is None else binary_bool_mask(
            fixed_mask, device=valid_mask.device, name="fixed_mask")
        if fixed.shape != valid_mask.shape or bool((fixed & ~valid_mask).any()):
            raise ValueError("fixed_mask must be a subset of valid_mask")
        if bool(fixed.any()):
            if initial_ids is None:
                raise ValueError("initial_ids are required for fixed positions")
            self._check_layout(fixed, initial_ids)
        cur = torch.full(valid_mask.shape, self.pad_token_id, device=valid_mask.device, dtype=torch.long)
        cur = cur.masked_fill(valid_mask, self.mask_token_id)
        if initial_ids is not None:
            if initial_ids.shape != valid_mask.shape:
                raise ValueError("initial_ids shape mismatch")
            cur = torch.where(fixed, initial_ids.long(), cur)
        history = []
        for step in range(steps):
            unknown = valid_mask & (cur == self.mask_token_id)
            if not bool(unknown.any()):
                break
            discrete_time, global_time = self._joint_time_from_input(cur, valid_mask)
            logits = self._logits(
                vl_embs_list, cur, valid_mask, state, vl_attention_mask,
                discrete_timestep=discrete_time, global_timestep=global_time,
                continuous_timestep=torch.zeros_like(discrete_time)).float()
            unknown_logits = logits[unknown]
            if decode_temperature == 0:
                selected = unknown_logits.argmax(-1)
                probs = unknown_logits.softmax(-1)
            else:
                probs = (unknown_logits / float(decode_temperature)).softmax(-1)
                selected = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            selected_prob = probs.gather(1, selected[:, None]).squeeze(1)
            if not bool(torch.isfinite(selected_prob).all()):
                raise RuntimeError("joint_maskgit received nonfinite categorical probabilities")
            proposed = cur.clone()
            proposed[unknown] = selected
            if step == steps - 1:
                next_unknown = torch.zeros_like(unknown)
            else:
                progress = (step + 1) / steps
                remaining = torch.floor(valid_counts.float() * math.cos(math.pi * progress / 2)).long()
                # Preserve exact counts, including ties and heterogeneous PAD.
                # For short rows, finish early rather than adding d=1 forwards.
                unknown_counts = unknown.flatten(1).sum(1)
                remaining = torch.minimum(remaining, (unknown_counts - 1).clamp_min(0))
                confidence = torch.full(valid_mask.shape, float("inf"), device=cur.device, dtype=torch.float32)
                score = selected_prob.clamp_min(torch.finfo(torch.float32).tiny).log()
                if choice_temperature > 0:
                    uniform = torch.rand(score.shape, device=cur.device, dtype=torch.float32,
                                         generator=generator).clamp_(1e-7, 1 - 1e-7)
                    gumbel = -torch.log(-torch.log(uniform))
                    score = score + float(choice_temperature) * (1 - progress) * gumbel
                confidence[unknown] = score
                ranks = confidence.flatten(1).argsort(dim=1, stable=True).argsort(dim=1, stable=True)
                next_unknown = (ranks < remaining[:, None]).reshape_as(unknown) & unknown
            cur = proposed.masked_fill(next_unknown, self.mask_token_id)
            if return_history:
                history.append(cur.clone())
        if bool(((cur == self.mask_token_id) & valid_mask).any()):
            raise RuntimeError("joint_maskgit terminated with unresolved valid MASK tokens")
        output = torch.where(valid_mask, cur, -1)
        self._check_layout(valid_mask, output)
        return (output, history) if return_history else output

    def forward_bd3lm(self, vl_embs_list, token_ids, valid_mask, state=None, vl_attention_mask=None):
        """One-forward BD3LM training for the two RVQ blocks.

        ``token_ids`` are the clean targets.  The noisy copy starts fully
        masked and is followed by the clean copy as context.  Both copies use
        the inference action positions, so RVQ2 reads RVQ1+G at the indices the
        coarse tokens occupy after they are written back.  Only logits from the
        noisy copy contribute to CE, and the attention mask prevents block 0
        from reading either clean or noisy block 1.
        """
        valid_mask = self._check_layout(valid_mask, token_ids)
        if not bool(valid_mask.any()):
            raise ValueError("BD3LM requires at least one valid token")
        noisy = torch.where(valid_mask, torch.full_like(token_ids, self.mask_token_id), self.pad_token_id)
        logits = self._logits(
            vl_embs_list, noisy, valid_mask, state, vl_attention_mask,
            global_timestep=torch.zeros(token_ids.shape[0], dtype=torch.long, device=token_ids.device),
            discrete_timestep=torch.zeros(token_ids.shape[0], dtype=torch.long, device=token_ids.device),
            continuous_timestep=torch.zeros(token_ids.shape[0], dtype=torch.long, device=token_ids.device),
            clean_token_ids=token_ids,
        )
        ce = F.cross_entropy(logits[valid_mask].float(), token_ids[valid_mask].long(), reduction="none")
        row_ids = torch.arange(token_ids.shape[0], device=token_ids.device)[:, None, None].expand_as(token_ids)[valid_mask]
        sums = torch.zeros(token_ids.shape[0], device=token_ids.device, dtype=ce.dtype).scatter_add(0, row_ids, ce)
        counts = valid_mask.flatten(1).sum(1).clamp_min(1)
        loss = (sums / counts).mean()
        return {"action_loss": loss, "logits": logits, "loss_mask": valid_mask}

    def forward(self, vl_embs_list, token_ids, valid_mask, state=None, vl_attention_mask=None, generator=None,
                phase_timestep=None, temporal_stage=None):
        if self.joint_maskgit:
            if temporal_stage is not None:
                raise ValueError("joint_maskgit has no RVQ stage")
            return self.forward_joint_maskgit(
                vl_embs_list, token_ids, valid_mask, state, vl_attention_mask, generator)
        valid_mask = self._check_layout(valid_mask, token_ids)
        if not valid_mask.any():
            raise ValueError("A training batch must contain at least one valid target token")
        block_ar_training = False
        if (self.temporal_decode and self.temporal_train
                and temporal_stage is not None
                and self.temporal_train_mode == "block_ar"):
            masked, loss_mask, temporal_timestep = self.apply_stagewise_mask(
                token_ids, valid_mask, stage_override=temporal_stage)
            discrete_timestep = temporal_timestep
            block_ar_training = True
        elif self.temporal_decode and self.temporal_train and self.temporal_train_mode == "block_ar":
            masked, loss_mask, temporal_timestep = self.apply_block_ar_mask(
                token_ids, valid_mask, phase_timestep=phase_timestep)
            discrete_timestep = temporal_timestep
            block_ar_training = True
        elif self.temporal_decode and self.temporal_train:
            if self.temporal_train_mode == "bd3lm":
                raise RuntimeError("BD3LM training must call forward_bd3lm")
            raise RuntimeError("temporal training must be block_ar or bd3lm")
        else:
            # Full MASK at t=0. A caller-supplied diffusion phase must not move
            # AdaLN away from the single argmax used at inference.
            phase_timestep = torch.zeros(
                token_ids.shape[0], device=token_ids.device, dtype=torch.float32)
            masked, loss_mask, discrete_timestep = self.apply_mask(
                token_ids, valid_mask, return_timestep=True,
                timestep_override=phase_timestep)
        # DF uses three time streams. During the discrete half, global t is
        # in [0,.5], the discrete branch runs at 2t, and the continuous
        # branch is held at its source endpoint.
        if phase_timestep is not None:
            global_timestep = (phase_timestep * self.num_timestep_buckets).long().clamp_(0, self.num_timestep_buckets - 1)
            discrete_timestep = (2.0 * phase_timestep * self.num_timestep_buckets).long().clamp_(0, self.num_timestep_buckets - 1)
            continuous_timestep = torch.zeros_like(discrete_timestep)
        else:
            # apply_mask samples a normalized ratio in [0, 1]. DF converts
            # that ratio to the integer timestep buckets before AdaLN.
            discrete_timestep = (discrete_timestep * self.num_timestep_buckets).long().clamp_(
                0, self.num_timestep_buckets - 1)
            global_timestep = discrete_timestep
            continuous_timestep = discrete_timestep
        logits = self._logits(vl_embs_list, masked, valid_mask, state, vl_attention_mask,
                              global_timestep=global_timestep,
                              discrete_timestep=discrete_timestep,
                              continuous_timestep=continuous_timestep,
                              block_ar_training=block_ar_training)
        ce = F.cross_entropy(logits[loss_mask].float(), token_ids[loss_mask].long(), reduction="none")
        # Equal per-example weighting, matching upstream, with empty examples excluded.
        row_ids = torch.arange(token_ids.shape[0], device=token_ids.device)[:, None, None].expand_as(token_ids)[loss_mask]
        sums = torch.zeros(token_ids.shape[0], device=token_ids.device, dtype=ce.dtype).scatter_add(0, row_ids, ce)
        counts = loss_mask.flatten(1).sum(1)
        active = valid_mask.flatten(1).any(1)
        # A valid example whose corruption was entirely cancelled contributes
        # zero, as upstream does; truly empty padded examples remain excluded.
        loss = (sums[active] / counts[active].clamp_min(1)).mean()
        if not bool(loss_mask.any()):
            loss = logits[..., 0][valid_mask].sum() * 0
        return {"action_loss": loss, "logits": logits, "loss_mask": loss_mask}

    def _fill_argmax(self, vl_embs_list, token_ids, valid_mask, unknown, state, vl_attention_mask, *, kv_cache=None):
        """Replace unknown tokens with one argmax. CE training uses the same t=0."""
        zeros = torch.zeros(token_ids.shape[0], device=token_ids.device, dtype=torch.long)
        logits = self._logits(
            vl_embs_list, token_ids, valid_mask, state, vl_attention_mask,
            kv_cache=kv_cache,
            global_timestep=zeros, discrete_timestep=zeros, continuous_timestep=zeros,
        ).float()
        sampled = logits.argmax(-1)
        return torch.where(unknown, sampled, token_ids)

    @torch.no_grad()
    def _predict_masked_subset(
        self, vl_embs_list, valid_mask, base_ids, active_mask, fixed_mask,
        state=None, vl_attention_mask=None, decode_temperature=0.1,
        choice_temperature=0.1, steps=1, generator=None, remask_enabled=False,
        remask_threshold=0.5, return_history=False, early_stop=True,
        kv_cache=None,
    ):
        """Decode active slots with one argmax. Other unresolved slots stay masked."""
        del decode_temperature, choice_temperature, steps, generator, remask_enabled, remask_threshold, early_stop
        cur = base_ids.clone()
        cur = torch.where(valid_mask, cur, self.pad_token_id)
        cur = torch.where(active_mask & ~fixed_mask, self.mask_token_id, cur)
        eligible = active_mask & valid_mask & ~fixed_mask
        if bool(eligible.any()):
            cur = self._fill_argmax(
                vl_embs_list, cur, valid_mask, eligible, state, vl_attention_mask, kv_cache=kv_cache,
            )
        if ((cur == self.mask_token_id) & eligible).any():
            raise RuntimeError("Temporal block stage ended with unresolved MASK tokens")
        history = [cur.clone()] if return_history else []
        return cur, history

    @torch.no_grad()
    def _predict_bd3lm_two_pass(
        self, vl_embs_list, valid_mask, state=None, vl_attention_mask=None,
        *, initial_ids=None, fixed_mask=None, decode_temperature=0.0,
        choice_temperature=0.0, return_history=False,
    ):
        """Run the two fixed RVQ forwards used by BD3LM training.

        Each forward uses the doubled [noisy, clean] action layout that
        forward_bd3lm trains on. The first pass predicts RVQ1+G from a
        masked noisy copy. Its detached predictions are then written into
        both the noisy copy and the clean context of the second pass; the
        second pass predicts RVQ2+ from the remaining masked slots. Logits
        are always taken from the noisy (first) copy.
        """
        valid_mask = self._check_layout(valid_mask)
        if not self.temporal_decode or self.temporal_train_mode != "bd3lm":
            raise RuntimeError("BD3LM two-pass decode requires temporal_decode and train_mode=bd3lm")
        fixed = (
            torch.zeros_like(valid_mask) if fixed_mask is None
            else binary_bool_mask(fixed_mask, device=valid_mask.device, name="fixed_mask")
        )
        if fixed.shape != valid_mask.shape or bool((fixed & ~valid_mask).any()):
            raise ValueError("fixed_mask must be a subset of valid_mask")
        cur = torch.full(
            valid_mask.shape, self.mask_token_id,
            device=valid_mask.device, dtype=torch.long,
        )
        cur = torch.where(valid_mask, cur, self.pad_token_id)
        if initial_ids is not None:
            if tuple(initial_ids.shape) != tuple(cur.shape):
                raise ValueError("initial_ids shape mismatch")
            cur = torch.where(fixed, initial_ids.to(device=cur.device, dtype=torch.long), cur)

        coarse, refine = self._temporal_stage_masks(valid_mask)
        history = []
        zeros = torch.zeros(
            valid_mask.shape[0], device=valid_mask.device, dtype=torch.long
        )
        # No stochastic MaskGIT schedule is used here: both forwards are
        # fixed-timestep t=0 passes, exactly as in forward_bd3lm.
        for stage_index, stage_mask in enumerate((coarse, refine)):
            unresolved = stage_mask & ~fixed
            noisy = torch.where(valid_mask, cur, self.pad_token_id)
            # Clean context contains the already-decoded earlier stage. At
            # stage 0 it is all MASK; at stage 1 predicted coarse tokens
            # occupy the clean half at the same action positions as training.
            clean = torch.full_like(cur, self.mask_token_id)
            clean = torch.where(valid_mask, clean, self.pad_token_id)
            clean = torch.where(fixed & valid_mask, cur.detach(), clean)
            if stage_index == 1:
                clean = torch.where(coarse, cur.detach(), clean)
            logits = self._logits(
                vl_embs_list, noisy, valid_mask, state, vl_attention_mask,
                global_timestep=zeros, discrete_timestep=zeros,
                continuous_timestep=zeros, clean_token_ids=clean,
            ).float()
            sampled = logits.argmax(-1)
            cur = torch.where(unresolved, sampled, cur)
            cur = torch.where(valid_mask, cur, self.pad_token_id)
            if return_history:
                history.append(torch.where(valid_mask, cur, -1).clone())
        output = torch.where(valid_mask, cur, -1)
        return (output, history) if return_history else output

    @torch.no_grad()
    def _predict_temporal(
        self, vl_embs_list, valid_mask, state=None, vl_attention_mask=None,
        *, initial_ids=None, fixed_mask=None, decode_temperature=0.1,
        choice_temperature=0.1, num_inference_steps=None, generator=None,
        return_history=False, early_stop=True,
        secondary_remask=None, secondary_remask_threshold=None,
    ):
        valid_mask = self._check_layout(valid_mask)
        steps = int(self.num_inference_steps if num_inference_steps is None else num_inference_steps)
        if steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        if not isinstance(early_stop, bool):
            raise ValueError("early_stop must be boolean")
        steps = 1
        decode_temperature = choice_temperature = 0.0
        if not all(math.isfinite(float(x)) and float(x) >= 0 for x in (decode_temperature, choice_temperature)):
            raise ValueError("decode and choice temperatures must be >= 0")
        remask_enabled = False
        remask_threshold = (
            self.secondary_remask_threshold if secondary_remask_threshold is None
            else float(secondary_remask_threshold)
        )
        if not math.isfinite(remask_threshold) or not 0 <= remask_threshold <= 1:
            raise ValueError("secondary_remask_threshold must be finite in [0, 1]")
        fixed = (
            torch.zeros_like(valid_mask) if fixed_mask is None
            else binary_bool_mask(fixed_mask, device=valid_mask.device, name="fixed_mask")
        )
        if fixed.shape != valid_mask.shape or bool((fixed & ~valid_mask).any()):
            raise ValueError("fixed_mask must be a subset of valid_mask")
        if fixed.any():
            if initial_ids is None:
                raise ValueError("initial_ids are required for fixed positions")
            self._check_layout(fixed, initial_ids)
        cur = torch.full(valid_mask.shape, self.mask_token_id, device=valid_mask.device, dtype=torch.long)
        cur = torch.where(valid_mask, cur, self.pad_token_id)
        if initial_ids is not None:
            if tuple(initial_ids.shape) != tuple(cur.shape):
                raise ValueError("initial_ids shape mismatch")
            cur = torch.where(fixed, initial_ids.long(), cur)
        resolved = fixed.clone()
        coarse, refine = self._temporal_stage_masks(valid_mask)
        time = torch.arange(self.action_horizon, device=valid_mask.device)[None, :, None]
        history = []
        # Static VLM K/V are shared by all MaskGIT steps and temporal blocks.
        # Hybrid DF has branch-specific global/discrete timestep streams.
        # The temporal KV path executes the compatibility transformer
        # property with one timestep and would silently erase that separation;
        # disable it for HybridActionDiT until a branch-aware cache exists.
        use_temporal_cache = self.temporal_kv_cache and not getattr(self.model, "is_hybrid", False)
        kv_cache = (
            {"temporal": True, "cross": [None] * self.num_layers}
            if use_temporal_cache else None
        )
        if kv_cache is not None:
            self._init_temporal_kv_cache(
                kv_cache, vl_embs_list, cur, valid_mask, state, vl_attention_mask
            )
        num_blocks = (self.action_horizon + self.temporal_block_size - 1) // self.temporal_block_size
        for block in range(num_blocks):
            start = block * self.temporal_block_size
            end = min(self.action_horizon, start + self.temporal_block_size)
            in_block = (time >= start) & (time < end)
            for stage_mask in (coarse, refine):
                active = valid_mask & in_block & stage_mask
                if bool((active & ~resolved).any()):
                    if kv_cache is not None:
                        kv_cache["block_index"] = block
                        kv_cache["stage"] = 0 if stage_mask is coarse else 1
                    cur, stage_history = self._predict_masked_subset(
                        vl_embs_list, valid_mask, cur, active, resolved,
                        state=state, vl_attention_mask=vl_attention_mask,
                        decode_temperature=decode_temperature,
                        choice_temperature=choice_temperature, steps=steps,
                        generator=generator, remask_enabled=remask_enabled,
                        remask_threshold=remask_threshold,
                        return_history=return_history, early_stop=early_stop,
                        kv_cache=kv_cache,
                    )
                    if kv_cache is not None:
                        self._commit_temporal_stage(kv_cache)
                    resolved = resolved | active
                    if return_history:
                        history.extend(stage_history)
                else:
                    resolved = resolved | active
        if bool((cur == self.mask_token_id)[valid_mask].any()):
            raise RuntimeError("Temporal decode terminated with unresolved valid MASK tokens")
        output = torch.where(valid_mask, cur, -1)
        return (output, history) if return_history else output

    @torch.no_grad()
    def predict(
        self, vl_embs_list, valid_mask, state=None, vl_attention_mask=None,
        *, initial_ids=None, fixed_mask=None, decode_temperature=None,
        choice_temperature=None, num_inference_steps=None, generator=None,
        return_history=False, early_stop=True,
        secondary_remask=None, secondary_remask_threshold=None,
    ):
        if decode_temperature is None:
            decode_temperature = 1.0 if self.joint_maskgit else 0.1
        if choice_temperature is None:
            choice_temperature = 1.0 if self.joint_maskgit else 0.1
        if not all(math.isfinite(t) and t >= 0 for t in (decode_temperature, choice_temperature)):
            raise ValueError("decode and choice temperatures must be >= 0")
        if not isinstance(early_stop, bool):
            raise ValueError("early_stop must be boolean")
        if self.joint_maskgit:
            if secondary_remask:
                raise ValueError("joint_maskgit uses committed locking without secondary remasking")
            return self._predict_joint_maskgit(
                vl_embs_list, valid_mask, state, vl_attention_mask,
                initial_ids=initial_ids, fixed_mask=fixed_mask,
                decode_temperature=decode_temperature, choice_temperature=choice_temperature,
                num_inference_steps=num_inference_steps, generator=generator,
                return_history=return_history)
        remask_enabled = False
        remask_threshold = (self.secondary_remask_threshold
                            if secondary_remask_threshold is None
                            else float(secondary_remask_threshold))
        if (not math.isfinite(remask_threshold) or not 0.0 <= remask_threshold <= 1.0):
            raise ValueError("secondary_remask_threshold must be finite in [0, 1]")
        valid_mask = self._check_layout(valid_mask)
        del remask_enabled, remask_threshold, num_inference_steps, generator, early_stop
        fixed = (torch.zeros_like(valid_mask) if fixed_mask is None else
                 binary_bool_mask(fixed_mask, device=valid_mask.device, name="fixed_mask"))
        if fixed.shape != valid_mask.shape or (fixed & ~valid_mask).any():
            raise ValueError("fixed_mask must be a subset of valid_mask")
        if fixed.any():
            if initial_ids is None:
                raise ValueError("initial_ids are required for fixed positions")
            self._check_layout(fixed, initial_ids)
        cur = torch.full(valid_mask.shape, self.mask_token_id, device=valid_mask.device, dtype=torch.long)
        cur = torch.where(valid_mask, cur, self.pad_token_id)
        if initial_ids is not None:
            if initial_ids.shape != cur.shape:
                raise ValueError("initial_ids shape mismatch")
            cur = torch.where(fixed, initial_ids.long(), cur)
        unknown = valid_mask & ~fixed
        if bool(unknown.any()):
            cur = self._fill_argmax(
                vl_embs_list, cur, valid_mask, unknown, state, vl_attention_mask,
            )
        if ((cur == self.mask_token_id) & valid_mask).any():
            raise RuntimeError("Decode terminated with unresolved valid MASK tokens")
        output = torch.where(valid_mask, cur, -1)
        history = [cur.clone()] if return_history else []
        return (output, history) if return_history else output

    predict_action = predict

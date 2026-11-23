"""QwenPI-v3 continuous policy: Qwen VLM, per-layer projectors, flow-matching DiT.

The module graph matches starVLA QwenPI_v3. Proprioception is written into the
instruction as text bins. Invalid joint slots stay out of the flow loss. There
is no discrete tokenizer and no valid-mask embedding.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf

from robomorpher.model.architectures.qwen_discrete_forcing import (
    _action_expert_autocast, _boolean, _normalize_framework_config, _positive_int,
    resolve_execute_horizon,
)
from robomorpher.model.backbones.action.continuous_action_head import ContinuousActionHead
from robomorpher.model.backbones.vlm.qwen_interface import QwenVLBackbone
from robomorpher.model.tokenizer.action_codec import ContinuousActionCodec
from robomorpher.model.tokenizer.state_tokenizer import JointStateTokenizer
from robomorpher.utils.model.masking import binary_bool_mask


class QwenPI_v3(nn.Module):
    """Assemble the VLM, layer projectors, and the continuous action DiT."""

    def __init__(self, config, *, codec=None, backbone=None):
        super().__init__()
        cfg = _normalize_framework_config(config, require_action_tokenizer=False)
        if cfg.framework.name not in {"QwenPI_v3", "QwenDiscreteDiffusion"}:
            raise ValueError("framework.name must be QwenPI_v3")
        self.config = cfg
        if "datasets" not in cfg:
            cfg.datasets = {"policy_data": {}}
        elif "policy_data" not in cfg.datasets:
            cfg.datasets.policy_data = {}
        ac = cfg.framework.action_model
        if str(ac.action_representation) != "continuous":
            raise ValueError("QwenPI_v3 requires action_representation=continuous")
        self.action_representation = "continuous"
        self.action_type = str(cfg.get("data", {}).get("action_type", "joint")).lower()
        if self.action_type != "joint":
            raise ValueError("data.action_type must be joint")
        if codec is not None and not isinstance(codec, ContinuousActionCodec):
            raise ValueError("QwenPI_v3 cannot use a discrete tokenizer codec")
        if codec is None:
            normalization = str(ac.get("continuous_normalization", "q99"))
            if normalization not in {"q99", "tokenizer_stats", "minmax"}:
                raise ValueError("continuous_normalization must be q99, tokenizer_stats, or minmax")
            stats_path = ac.get("continuous_stats_path")
            if normalization in {"q99", "minmax"} and not stats_path:
                data_root = cfg.get("data", {}).get("root")
                if not data_root:
                    raise ValueError(f"{normalization} requires data.root or action_model.continuous_stats_path")
                stats_path = str(Path(data_root) / "meta" / "stats_masked.json")
            if stats_path and not Path(stats_path).is_absolute():
                stats_path = str((Path(__file__).resolve().parents[3] / stats_path).resolve())
            self.codec = ContinuousActionCodec(
                num_arms=int(ac.get("num_arms", 2)),
                max_num_joints=int(ac.get("max_num_joints", 7)),
                angle_low=float(ac.get("angle_low", -2 * math.pi)),
                angle_high=float(ac.get("angle_high", 2 * math.pi)),
                normalization=normalization, stats_path=stats_path,
                value_low=ac.get("value_low"), value_high=ac.get("value_high"),
                action_representation=self.action_type,
            )
        else:
            self.codec = codec
        self.codec.requires_grad_(False)
        self.backbone = backbone if backbone is not None else QwenVLBackbone(cfg)
        self._build(cfg, ac)
        self.codec.eval()

    def _build(self, cfg, ac):
        del cfg
        self.repeated_diffusion_steps = _positive_int(
            ac.get("repeated_diffusion_steps", 4), "repeated_diffusion_steps")
        self._explicit_state_encoding = "state_tokenizer" in ac
        self.action_horizon = _positive_int(ac.get("action_horizon", 50), "action_horizon")
        self.execute_horizon = resolve_execute_horizon(ac, self.action_horizon)
        self.discrete_action_horizon = self.action_horizon
        self.num_layers = _positive_int(
            self.backbone.num_layers if ac.get("num_layers") is None else ac.num_layers, "num_layers")
        if self.num_layers > self.backbone.num_layers:
            raise ValueError("Action head layers exceed available VLM feature layers")
        hidden = _positive_int(
            self.backbone.hidden_size if ac.get("hidden_dim") is None else ac.hidden_dim, "hidden_dim")
        dit_cfg = dict(ac.get("diffusion_model_cfg", {}))
        dit_cfg.setdefault("interleave_self_attention", True)
        dit_cfg.setdefault("positional_embeddings", None)
        head_dim = _positive_int(dit_cfg.get("attention_head_dim", 64), "diffusion_model_cfg.attention_head_dim")
        heads = _positive_int(max(1, hidden // head_dim) if ac.get("num_heads") is None else ac.num_heads, "num_heads")
        if hidden % heads or ("attention_head_dim" in dit_cfg and hidden != heads * head_dim):
            raise ValueError("hidden_dim must equal num_heads * attention_head_dim")
        self.include_urdf = _boolean(ac.get("urdf_conditioning", False), "urdf_conditioning")
        self.include_state = _boolean(ac.get("state_conditioning", True), "state_conditioning")
        self.use_urdf_encoder = False
        self.head_include_state = False
        state_cfg = ac.get("state_tokenizer", {"mode": "text_bins"})
        if not isinstance(state_cfg, Mapping):
            raise ValueError("action_model.state_tokenizer must be a mapping")
        state_cfg = dict(state_cfg)
        self.state_mode = state_cfg.pop("mode", "text_bins")
        if self.state_mode not in ("text_bins", "continuous"):
            raise ValueError("state_tokenizer.mode must be text_bins or continuous")
        self.state_tokenizer = None
        self._state_encoding = {"mode": self.state_mode, "enabled": self.include_state,
                                "urdf_conditioning": self.include_urdf}
        if self.state_mode == "text_bins":
            unknown = set(state_cfg) - {"num_bins", "text_format", "pad_id", "angle_wrap", "angle_low", "angle_high",
                                        "value_low", "value_high", "normalization", "state_stats_path", "state_stats"}
            if unknown:
                raise ValueError(f"Unknown text state settings: {sorted(unknown)}")
            from robomorpher.model.tokenizer.state_tokenizer import instruction_state_quantiles, reject_fixed_state_bounds
            reject_fixed_state_bounds(state_cfg)
            pinned = state_cfg.pop("state_stats", None)
            path = state_cfg.pop("state_stats_path", None) or getattr(self.codec, "stats_path", None)
            state_cfg["state_stats"] = instruction_state_quantiles(
                action_type=self.action_type, num_arms=self.codec.num_arms,
                max_num_joints=self.codec.max_num_joints, stats_path=path, state_stats=pinned)
            tokenizer = JointStateTokenizer(**state_cfg, normalization="state_q01_q99")
            self._state_encoding["tokenizer"] = tokenizer.metadata()
            self._state_encoding["num_arms"] = self.codec.num_arms
            self._state_encoding["max_num_joints"] = self.codec.max_num_joints
            if self.include_state:
                self.state_tokenizer = tokenizer
        self.head_include_state = self.include_state and self.state_mode == "continuous"
        state_dim = self.codec.state_condition_dim if (self.include_urdf or self.head_include_state) else 0
        expected_dim = self.codec.action_dim
        if ac.get("action_dim") is not None and int(ac.action_dim) != expected_dim:
            raise ValueError(f"action_dim must equal continuous padded layout {expected_dim}")
        # starVLA: one LayerNorm+Linear per VLM layer when the DiT is narrower,
        # otherwise Identity. Self-attention layers ignore the projected states.
        vl_width = int(self.backbone.hidden_size)
        self.project_layers = nn.ModuleList([
            nn.Identity() if vl_width == hidden else nn.Sequential(
                nn.LayerNorm(vl_width), nn.Linear(vl_width, hidden))
            for _ in range(self.num_layers)
        ])
        self.action_model = ContinuousActionHead(
            action_dim=expected_dim, action_horizon=self.action_horizon,
            vl_hidden_dim=hidden, num_layers=self.num_layers,
            hidden_dim=hidden, num_heads=heads, state_dim=state_dim,
            future_tokens=_positive_int(ac.get("future_tokens", 32), "future_tokens", minimum=0),
            max_seq_len=ac.get("max_seq_len"),
            add_pos_embed=ac.get("add_pos_embed", True),
            dropout=float(ac.get("dropout", .2)), diffusion_model_cfg=dit_cfg,
            loss_type=str(ac.get("continuous_loss", "l2")),
            num_inference_steps=_positive_int(ac.get("num_inference_steps", 4), "num_inference_steps"),
            flow_matching_style="starvla",
            noise_beta_alpha=float(ac.get("noise_beta_alpha", 1.5)),
            noise_beta_beta=float(ac.get("noise_beta_beta", 1.0)),
            noise_s=float(ac.get("noise_s", 0.999)),
            num_timestep_buckets=int(ac.get("num_timestep_buckets", 1000)),
            valid_mask_embedding=bool(ac.get("valid_mask_embedding", False)))
        if ac.get("gradient_checkpointing", True):
            self.action_model.gradient_checkpointing_enable()
        self._diffusion_contract = {
            "action_representation": "continuous", "flow_matching": True,
            "action_type": self.action_type,
            "ee_pose": "xyz_rot6d" if self.action_type == "ee" else None,
            "repeated_diffusion_steps": int(self.repeated_diffusion_steps),
            "continuous_loss": self.action_model.loss_type,
            "action_token_layout": "one_temporal_token_per_timestep",
            "action_encoder": "qwenpi_v3_action_mlp_with_sinusoidal_time",
            "dit_output": "pre_output",
            "block_positional_embeddings": None,
            "valid_mask_embedding": bool(self.action_model.valid_mask_embedding_enabled),
            "project_layers": "identity" if vl_width == hidden else "layernorm_linear",
            "normalization": self.codec.normalization_metadata(),
            "num_inference_steps": self.action_model.num_inference_steps,
            "hidden_dim": hidden, "num_heads": heads,
            "state_dim": state_dim, "action_dim": expected_dim,
        }

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval()
        return self

    def _backbone_examples(self, examples):
        if self.state_tokenizer is None:
            return examples
        with torch.autocast(device_type=self.state_tokenizer.device.type, enabled=False):
            encoded = self.state_tokenizer.encode_examples(
                examples, num_arms=self.codec.num_arms, max_num_joints=self.codec.max_num_joints)
            instructions = self.state_tokenizer.format_instructions(
                [example["lang"] for example in examples], encoded)
        return [dict(example, lang=instruction) for example, instruction in zip(examples, instructions)]

    def _encode_vl_features(self, examples):
        states, mask = self.backbone(self._backbone_examples(examples), self.num_layers)
        if len(states) != len(self.project_layers):
            raise ValueError("VLM layer count does not match project_layers")
        mask = binary_bool_mask(mask, device=states[0].device, name="vl_attention_mask")
        projected = []
        for projector, hidden in zip(self.project_layers, states):
            hidden = torch.where(mask[..., None], hidden, torch.zeros_like(hidden))
            weight = next(projector.parameters(), None)
            if weight is not None:
                hidden = hidden.to(dtype=weight.dtype)
            hidden = projector(hidden)
            projected.append(torch.where(mask[..., None], hidden, torch.zeros_like(hidden)))
        return projected, mask

    def _condition(self, examples):
        if not (self.head_include_state or self.include_urdf):
            return None
        with torch.autocast(device_type=self.codec.device.type, enabled=False):
            return self.codec.state_condition(
                examples, include_urdf=self.include_urdf, include_state=self.head_include_state)

    def _repeat(self, states, actions, valid, vl_mask, state, examples):
        repeats = self.repeated_diffusion_steps
        if repeats <= 1:
            return states, actions, valid, vl_mask, state, examples
        states = [h.repeat(repeats, 1, 1) if h is not None else None for h in states]
        actions = actions.repeat(repeats, 1, 1)
        valid = valid.repeat(repeats, 1, 1)
        vl_mask = vl_mask.repeat(repeats, 1)
        if state is not None:
            if isinstance(state, Mapping):
                state = {k: (v.repeat(repeats, *([1] * (v.ndim - 1))) if v is not None else None)
                         for k, v in state.items()}
            else:
                state = state.repeat(repeats, *([1] * (state.ndim - 1)))
        return states, actions, valid, vl_mask, state, list(examples) * repeats

    def forward(self, examples):
        with torch.autocast(device_type=self.codec.device.type, enabled=False):
            actions, valid = self.codec.continuous_targets(examples)
        if actions.shape[1] != self.action_horizon:
            raise ValueError("Dataset action_chunk_size must equal action_model.action_horizon")
        states, vl_mask = self._encode_vl_features(examples)
        state = self._condition(examples)
        states, actions, valid, vl_mask, state, _ = self._repeat(
            states, actions, valid, vl_mask, state, examples)
        with _action_expert_autocast(states):
            return self.action_model(states, actions, valid, state=state, vl_attention_mask=vl_mask)

    @torch.inference_mode()
    def predict_action(self, examples, *, num_inference_steps=None, **_ignored):
        if not isinstance(examples, list):
            examples = [examples]
        valid = self.codec.valid_tokens(examples, self.action_horizon)
        states, vl_mask = self._encode_vl_features(examples)
        with _action_expert_autocast(states):
            normalized = self.action_model.predict(
                states, valid, state=self._condition(examples), vl_attention_mask=vl_mask,
                num_inference_steps=num_inference_steps)
        with torch.autocast(device_type=self.codec.device.type, enabled=False):
            result = self.codec.decode_continuous(normalized, examples, valid)
        result["action_decode_mode"] = "continuous_flow"
        result["normalized_actions"] = normalized.float().cpu().numpy()
        return result

    def checkpoint_contract(self):
        return {
            "framework": self.config.framework.name,
            "action_representation": "continuous",
            "codec": self.codec.spec.to_dict(),
            "action_horizon": self.action_horizon,
            "num_layers": self.num_layers,
            "vl_hidden_dim": self.backbone.hidden_size,
            "backbone_id": self.config.framework.qwenvl.base_vlm,
            "backbone_family": getattr(self.backbone, "family", None),
            "backbone_revision": getattr(self.backbone, "hf_revision", None),
            "continuous_flow": self._diffusion_contract,
            "state_encoding": self._state_encoding if self._explicit_state_encoding else None,
        }

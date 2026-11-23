"""Opt-in FAST autoregression followed by the existing physical flow refiner.

FAST positions describe a compressed whole chunk, rather than time/joint/RVQ
slots. Only this module handles that sequence layout; the original frame-codec
path keeps its own config, masks and checkpoint contract. Both heads still use
one HybridActionDiT owner, and the continuous branch only receives hard-decoded
physical actions, never teacher-forced FAST hidden states or token embeddings.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robomorpher.model.backbones.action.continuous_action_head import ContinuousActionHead
from robomorpher.model.backbones.action.fast_autoregressive_head import FastAutoregressiveHead
from robomorpher.model.backbones.action.hybrid_action_dit import HybridActionDiT
from robomorpher.model.tokenizer.state_tokenizer import JointStateTokenizer


def _positive(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _bool(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _finite(value, name, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return result


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def init_fast_refinement(self, cfg, ac):
    """Build only the explicitly selected FAST + AR + DF configuration."""
    if self.action_representation != "discrete":
        raise ValueError("FAST requires action_representation=discrete")
    fast_cfg = _mapping(ac.get("fast_autoregressive", {}), "fast_autoregressive")
    if set(fast_cfg) - {"enabled"}:
        raise ValueError(f"Unknown FAST autoregressive settings: {sorted(set(fast_cfg) - {'enabled'})}")
    if not _bool(fast_cfg.get("enabled", True), "fast_autoregressive.enabled"):
        raise ValueError("FAST requires fast_autoregressive.enabled=true")
    if ac.get("backbone", "categorical_dit") != "categorical_dit":
        raise ValueError("FAST shared refinement requires categorical_dit")
    if bool(ac.get("maskgit_enabled", False)):
        raise ValueError("FAST autoregression cannot use MaskGIT")
    if ac.get("secondary_remask", False):
        raise ValueError("FAST autoregression does not support discrete diffusion remasking")
    temporal = _mapping(ac.get("temporal_decode", {}), "temporal_decode")
    if temporal.get("enabled", False) or temporal.get("train", False):
        raise ValueError("FAST sequence positions cannot use temporal/RVQ block decoding")
    self.repeated_diffusion_steps = _positive(ac.get("repeated_diffusion_steps", 2), "repeated_diffusion_steps")
    self.action_horizon = _positive(ac.get("action_horizon", 50), "action_horizon")
    from robomorpher.model.architectures.qwen_discrete_forcing import resolve_execute_horizon
    self.execute_horizon = resolve_execute_horizon(ac, self.action_horizon)
    self.discrete_action_horizon = _positive(ac.get("discrete_action_horizon", self.action_horizon), "discrete_action_horizon")
    if self.action_horizon != self.discrete_action_horizon or self.codec.action_horizon != self.action_horizon:
        raise ValueError("FAST encodes the full chunk: codec/action/discrete horizons must match (no 50->10 projection)")
    action_dim = int(self.codec.num_arms * (self.codec.max_num_joints + 1))
    if ac.get("action_dim", action_dim) != action_dim:
        raise ValueError("action_dim must match FAST padded joint-plus-gripper layout")
    for key, actual in (("num_arms", self.codec.num_arms), ("max_num_joints", self.codec.max_num_joints)):
        if ac.get(key, actual) != actual:
            raise ValueError(f"action_model.{key} conflicts with FAST codec")
    self.num_layers = _positive(self.backbone.num_layers if ac.get("num_layers") is None else ac.num_layers, "num_layers")
    if self.num_layers > self.backbone.num_layers:
        raise ValueError("Action head layers exceed available VLM feature layers")
    for key, actual in (("vl_hidden_dim", self.backbone.hidden_size), ("num_vl_layers", self.backbone.num_layers)):
        if cfg.framework.qwenvl.get(key) is not None and cfg.framework.qwenvl[key] != actual:
            raise ValueError(f"qwenvl.{key} conflicts with the actual backbone")
    hidden = _positive(self.backbone.hidden_size if ac.get("hidden_dim") is None else ac.hidden_dim, "hidden_dim")
    dit_cfg = _mapping(ac.get("diffusion_model_cfg", {}), "diffusion_model_cfg")
    dit_cfg.setdefault("interleave_self_attention", True)
    head_dim = _positive(dit_cfg.get("attention_head_dim", 64), "attention_head_dim")
    heads = _positive(max(1, hidden // head_dim) if ac.get("num_heads") is None else ac.num_heads, "num_heads")
    if hidden % heads or ("attention_head_dim" in dit_cfg and hidden != heads * head_dim):
        raise ValueError("hidden_dim must match num_heads * attention_head_dim")
    self.vl_projection = ac.get("vl_projection", "none")
    if self.vl_projection not in {"none", "layernorm_linear"}:
        raise ValueError("vl_projection must be none or layernorm_linear")
    cross_dim = hidden if self.vl_projection == "layernorm_linear" else self.backbone.hidden_size
    self.include_urdf = _bool(ac.get("urdf_conditioning", False), "urdf_conditioning")
    requested_urdf = ac.get("urdf_conditioning_mode", "raw_features")
    if requested_urdf not in {"auto", "raw_features"} or self.include_urdf:
        raise ValueError("FAST has no tokenizer URDF encoder; this branch requires urdf_conditioning=false and raw_features/auto mode")
    self.urdf_mode, self.update_urdf_tokens = "disabled", False
    self.use_urdf_encoder, self.urdf_adaln_conditioning = False, False
    self.include_state = _bool(ac.get("state_conditioning", True), "state_conditioning")
    self._explicit_state_encoding = "state_tokenizer" in ac
    state_cfg = _mapping(ac.get("state_tokenizer", {"mode": "text_bins"}), "state_tokenizer")
    self.state_mode = state_cfg.pop("mode", "text_bins")
    self.state_tokenizer = None
    self._state_encoding = {"mode": self.state_mode, "enabled": self.include_state,
                            "urdf_conditioning": False, "num_arms": self.codec.num_arms,
                            "max_num_joints": self.codec.max_num_joints}
    if self.state_mode == "text_bins":
        from robomorpher.model.tokenizer.state_tokenizer import instruction_state_quantiles, reject_fixed_state_bounds
        reject_fixed_state_bounds(state_cfg)
        pinned = state_cfg.pop("state_stats", None)
        path = state_cfg.pop("state_stats_path", None)
        if pinned is None and not path and cfg.get("data", {}).get("root"):
            path = str(Path(cfg.data.root) / "meta" / "stats_masked.json")
        state_cfg["state_stats"] = instruction_state_quantiles(
            action_type="joint", num_arms=self.codec.num_arms,
            max_num_joints=self.codec.max_num_joints, stats_path=path, state_stats=pinned)
        tokenizer = JointStateTokenizer(**state_cfg, normalization="state_q01_q99")
        self._state_encoding["tokenizer"] = tokenizer.metadata()
        if self.include_state:
            self.state_tokenizer = tokenizer
    elif self.state_mode == "continuous":
        if set(state_cfg) - {"num_bins", "text_format", "pad_id"}:
            raise ValueError("Unknown continuous state settings")
    else:
        raise ValueError("state_tokenizer.mode must be text_bins or continuous")
    self.head_include_state = self.include_state and self.state_mode == "continuous"
    state_dim = self.codec.state_condition_dim if self.head_include_state else 0
    if ac.get("state_dim", state_dim) != state_dim:
        raise ValueError("state_dim conflicts with FAST conditioning layout")
    dropout = _finite(ac.get("dropout", .2), "dropout")
    if dropout > 1:
        raise ValueError("dropout must be <= 1")
    self.decode_temperature = _finite(ac.get("decode_temperature", 0.0), "decode_temperature")
    self.choice_temperature = _finite(ac.get("choice_temperature", 0.0), "choice_temperature")
    if "num_inference_steps" in ac:
        raise ValueError("FAST generation length is the autoregressive token count, not num_inference_steps")
    self.decoder_adaln_trainable = False
    self.tokenizer_decoder_adaln = None
    self.rvq_stagewise_train = False
    future = _positive(ac.get("future_tokens", 32), "future_tokens", minimum=0)
    # One position table: continuous frames occupy 0..H-1, FAST tokens follow them.
    required_capacity = max(self.action_horizon + self.codec.max_tokens,
                            self.codec.max_tokens + int(state_dim > 0),
                            self.action_horizon + future + int(state_dim > 0))
    capacity = _positive(ac.get("max_seq_len", required_capacity), "max_seq_len")
    if capacity < required_capacity:
        raise ValueError("max_seq_len must hold both the FAST and continuous sequences")
    nested_capacity = dit_cfg.pop("max_num_positional_embeddings", capacity)
    if nested_capacity != capacity:
        raise ValueError("max_num_positional_embeddings must agree with max_seq_len")
    head_dit_cfg = dict(dit_cfg, max_num_positional_embeddings=capacity)
    self.action_model = FastAutoregressiveHead(
        vocab_size=self.codec.vocab_size, max_tokens=self.codec.max_tokens,
        vl_hidden_dim=cross_dim, num_layers=self.num_layers,
        hidden_dim=hidden, num_heads=heads, state_dim=state_dim,
        dropout=dropout, diffusion_model_cfg=head_dit_cfg,
        add_pos_embed=ac.get("add_pos_embed", True),
        bos_id=self.codec.bos_id, eos_id=self.codec.eos_id, pad_id=self.codec.pad_id)
    self.action_model.backbone_type = "fast_autoregressive"
    self.action_model.num_inference_steps = 1
    self.vl_projectors = nn.ModuleDict()
    if self.vl_projection == "layernorm_linear":
        self.vl_projectors.update({str(i): nn.Sequential(nn.LayerNorm(self.backbone.hidden_size),
                                                       nn.Linear(self.backbone.hidden_size, hidden))
                                   for i in range(self.num_layers)
                                   if not (self.action_model.interleave_self_attention and i % 2 == 1)})
    df = _mapping(ac.get("discrete_forcing", {}), "discrete_forcing")
    allowed = {"enabled", "num_shared_layers", "alpha", "inference_mode",
               "flow_loss_weight", "discrete_loss_weight", "paired_phase_sampling", "one_step_loss_prob",
               "one_step_loss_weight", "num_discrete_steps", "continuous_loss",
               "continuous_inference_steps", "continuous_stats_path"}
    if set(df) - allowed:
        raise ValueError(f"Unknown discrete_forcing settings: {sorted(set(df) - allowed)}")
    self.discrete_forcing_enabled = _bool(df.get("enabled", True), "discrete_forcing.enabled")
    if not self.discrete_forcing_enabled:
        raise ValueError("FAST refinement recipe requires discrete_forcing.enabled=true")
    shared = _positive(df.get("num_shared_layers", 2), "discrete_forcing.num_shared_layers")
    if shared >= self.num_layers:
        raise ValueError("num_shared_layers must be less than num_layers")
    if df.get("num_discrete_steps", 1) != 1:
        raise ValueError("FAST has one AR generation stage; num_discrete_steps must be 1")
    alpha = _finite(df.get("alpha", .3), "alpha")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0,1]")
    weights = {key: _finite(df.get(key, default), key) for key, default in (
        ("flow_loss_weight", 1.0), ("discrete_loss_weight", 1.0), ("one_step_loss_weight", 0.0), ("one_step_loss_prob", 0.0))}
    if weights["one_step_loss_prob"] > 1:
        raise ValueError("one_step_loss_prob must be <= 1")
    self.paired_phase_sampling = _bool(df.get("paired_phase_sampling", True), "paired_phase_sampling")
    if self.paired_phase_sampling and (self.repeated_diffusion_steps < 2 or self.repeated_diffusion_steps % 2):
        raise ValueError("paired_phase_sampling requires even repeated_diffusion_steps >= 2")
    self.discrete_forcing_inference_mode = df.get("inference_mode", "refine")
    if self.discrete_forcing_inference_mode not in {"discrete", "refine"}:
        raise ValueError("discrete_forcing.inference_mode must be discrete or refine")
    steps = _positive(df.get("continuous_inference_steps", 4), "continuous_inference_steps")
    stats_path = df.get("continuous_stats_path")
    if not stats_path and cfg.get("data", {}).get("root"):
        stats_path = str(Path(cfg.data.root) / "meta" / "stats_masked.json")
    if not stats_path:
        raise ValueError("FAST refinement needs continuous_stats_path or data.root")
    stats_path = Path(stats_path)
    if not stats_path.is_absolute():
        stats_path = (Path(__file__).resolve().parents[4] / stats_path).resolve()
    with stats_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    stats = raw.get("action", raw)
    if not isinstance(stats, Mapping) or "q01" not in stats or "q99" not in stats:
        raise ValueError("continuous stats require action.q01 and action.q99")
    q01, q99 = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
    if q01.shape != (action_dim,) or q99.shape != (action_dim,):
        raise ValueError(f"continuous stats must have shape {(action_dim,)}")
    if not np.isfinite(q01).all() or not np.isfinite(q99).all() or np.any(q99 <= q01):
        raise ValueError("continuous stats must be finite with q99 > q01")
    self.register_buffer("_df_q01", torch.as_tensor(q01), persistent=False)
    self.register_buffer("_df_q99", torch.as_tensor(q99), persistent=False)
    self.discrete_forcing_config = {**df, **weights, "enabled": True, "num_shared_layers": shared,
        "continuous_num_layers": self.num_layers, "continuous_hidden_dim": hidden,
        "continuous_num_heads": heads, "alpha": alpha, "continuous_source_alpha": alpha,
        "continuous_stats_path": str(stats_path),
        "detach_discrete_source": True,
        "continuous_inference_steps": steps, "num_continuous_steps": steps, "num_discrete_steps": 1,
        "paired_phase_sampling": self.paired_phase_sampling, "branch_input_mode": "full",
        "disjoint_branch_batch_routing": True, "isolate_branch_tokens_before_shared": True,
        "allow_branch_communication": False, "branch_communication_on_self_attention": False}
    self.action_model.model = HybridActionDiT.from_dit(
        self.action_model.model, shared,
        continuous_output_dim=hidden, discrete_output_dim=self.codec.output_vocab_size,
        conditioning_layout="df_per_token", future_tokens=future,
        max_seq_len=capacity, continuous_horizon=self.action_horizon)
    self.action_model.token_decoder = nn.Linear(hidden, self.action_model.output_vocab_size)
    nn.init.normal_(self.action_model.token_decoder.weight, std=0.02)
    nn.init.zeros_(self.action_model.token_decoder.bias)
    # The shared table owns positions. Drop the head-local copy.
    self.action_model.position_embedding = None
    self.continuous_refiner = ContinuousActionHead(
        discrete_mode="joint_maskgit",
        action_dim=action_dim, action_horizon=self.action_horizon, vl_hidden_dim=cross_dim,
        num_layers=self.num_layers, hidden_dim=hidden, num_heads=heads, state_dim=state_dim,
        future_tokens=future, max_seq_len=capacity, add_pos_embed=ac.get("add_pos_embed", True),
        dropout=dropout, diffusion_model_cfg=dit_cfg, loss_type=df.get("continuous_loss", "l2"),
        num_inference_steps=steps, flow_matching_style="starvla",
        noise_beta_alpha=float(ac.get("noise_beta_alpha", 1.5)), noise_beta_beta=float(ac.get("noise_beta_beta", 1.0)),
        noise_s=float(ac.get("noise_s", .999)), num_timestep_buckets=int(ac.get("num_timestep_buckets", 1000)),
        one_step_loss_prob=weights["one_step_loss_prob"], valid_mask_embedding=ac.get("valid_mask_embedding", True))
    # df_per_token already keeps global cross/self parity on the sliced stack.
    # Do not replace those blocks with the refiner's private DiT.
    self.continuous_refiner._modules.pop("model", None)
    object.__setattr__(self.continuous_refiner, "model", self.action_model.model.branch_view("continuous"))
    object.__setattr__(self.continuous_refiner, "hybrid_dit", self.action_model.model)
    self.continuous_refiner.future_tokens = None
    self.continuous_refiner.position_embedding = None
    if ac.get("gradient_checkpointing", True):
        self.action_model.gradient_checkpointing_enable()
        self.continuous_refiner.gradient_checkpointing_enable()
    self._fast_contract = {
        "version": 1, "token_layout": "whole_chunk_variable_length",
        "attention": "causal_teacher_forced_shift_right", "discrete_global_timestep": 0,
        "vocab_size": self.codec.vocab_size, "output_vocab_size": self.codec.output_vocab_size,
        "bos_id": self.codec.bos_id, "eos_id": self.codec.eos_id, "pad_id": self.codec.pad_id,
        "max_tokens_including_eos": self.codec.max_tokens, "hidden_dim": hidden, "num_heads": heads,
        "num_layers": self.num_layers, "state_dim": state_dim, "dropout": dropout,
        "add_pos_embed": ac.get("add_pos_embed", True), "diffusion_model_cfg": head_dit_cfg,
        "vl_projection": {"mode": self.vl_projection, "input_dim": self.backbone.hidden_size,
                          "output_dim": cross_dim, "layer_indices": [int(i) for i in self.vl_projectors]},
        "repeated_diffusion_steps": self.repeated_diffusion_steps, "decode_temperature": self.decode_temperature,
        "time_projection": "prefix_global_action_constant_discrete",
        "conditioning_layout": "df_per_token",
        "future_tokens": {"discrete": 0, "continuous": future},
        "continuous_refinement": {**self.discrete_forcing_config, "action_dim": action_dim,
            "architecture": "hybrid_dit_shared_trunk_v1", "continuous_conditioning": "decoded_physical_source_only",
            "continuous_source_train": "gt_token_decode", "continuous_source_inference": "predicted_token_decode",
            "normalization": "slot_q99_q01_clip_2.2", "stats_sha256": hashlib.sha256(q01.tobytes() + q99.tobytes()).hexdigest(),
            "continuous_loss": self.continuous_refiner.loss_type,
            "valid_mask_embedding": self.continuous_refiner.valid_mask_embedding_enabled,
            "noise_beta_alpha": self.continuous_refiner.noise_beta_alpha,
            "noise_beta_beta": self.continuous_refiner.noise_beta_beta, "noise_s": self.continuous_refiner.noise_s,
            "num_timestep_buckets": self.continuous_refiner.num_timestep_buckets,
            "max_seq_len": ac.get("max_seq_len"), "angle_low": ac.get("angle_low", -2 * math.pi),
            "angle_high": ac.get("angle_high", 2 * math.pi)}}
    self._fast_enabled = True
    self.codec.eval()


def _repeat(value, repeats):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_repeat(v, repeats) for v in value]
    if isinstance(value, Mapping):
        return {key: _repeat(v, repeats) for key, v in value.items()}
    return value.repeat(repeats, *([1] * (value.ndim - 1)))


def forward_fast(self, examples):
    if not isinstance(examples, list) or not examples:
        raise ValueError("FAST training requires a nonempty list of examples")
    for example in examples:
        if int(np.asarray(example["action_angle"]).shape[0]) != self.action_horizon:
            raise ValueError("Dataset action_chunk_size must equal FAST action_horizon")
    with torch.no_grad(), torch.autocast(device_type=self.codec.device.type, enabled=False):
        ids, valid = self.codec.encode_actions(examples)
    if ids.ndim != 2 or ids.shape != valid.shape or ids.shape[0] != len(examples):
        raise ValueError("FAST encoder must return whole-chunk ids/mask [B,L]")
    states, vl_mask = self._encode_vl_features(examples)
    state = self._condition(examples)
    # Paired routing repeats observations half for the teacher-forced AR branch
    # and half for flow training. The continuous call sees only decoded actions.
    repeats = self.repeated_diffusion_steps // 2 if self.paired_phase_sampling else self.repeated_diffusion_steps
    d_states, d_mask, d_state = _repeat(states, repeats), _repeat(vl_mask, repeats), _repeat(state, repeats)
    d_ids, d_valid = _repeat(ids, repeats), _repeat(valid, repeats)
    repeated_examples = examples * repeats
    from robomorpher.model.architectures.qwen_discrete_forcing import _action_expert_autocast
    with _action_expert_autocast(d_states):
        out = self.action_model(d_states, d_ids, d_valid, state=d_state, vl_attention_mask=d_mask)
    ce = out["action_loss"]
    c_phase = (self._df_phase_timestep(d_ids.shape[0], d_ids.device, continuous=True)
               if self.paired_phase_sampling else None)
    df = self._discrete_forcing_loss(d_states, d_mask, d_state, d_ids, d_valid,
                                    repeated_examples, continuous_examples=repeated_examples,
                                    phase_timestep=c_phase)
    weighted = self.discrete_forcing_config["flow_loss_weight"] * df["continuous_loss"]
    metrics = {"ce_loss": ce.detach(), "continuous_loss_weighted": weighted.detach(),
               "action_loss": self.discrete_forcing_config["discrete_loss_weight"] * ce + weighted,
               "fast_mean_tokens": valid.sum(-1).float().mean().detach()}
    for key in ("continuous_flow_loss", "continuous_one_step_loss", "continuous_loss"):
        metrics[key] = df[key].detach()
    if "token_accuracy" in out:
        metrics["token_accuracy"] = out["token_accuracy"].detach()
    return metrics


@torch.inference_mode()
def predict_fast(self, examples, states=None, vl_mask=None, condition=None, *,
                 decode_temperature=None, generator=None, return_history=False,
                 continuous_inference_steps=None, **kwargs):
    if kwargs:
        raise ValueError(f"Unsupported FAST prediction settings: {sorted(kwargs)}")
    if not isinstance(examples, list):
        examples = [examples]
    if not examples:
        raise ValueError("FAST prediction requires at least one observation")
    if (states is None) != (vl_mask is None):
        raise ValueError("Provide both states and vl_mask, or neither")
    if states is None:
        states, vl_mask = self._encode_vl_features(examples)
    if condition is None:
        condition = self._condition(examples)
    temperature = self.decode_temperature if decode_temperature is None else _finite(decode_temperature, "decode_temperature")
    from robomorpher.model.architectures.qwen_discrete_forcing import _action_expert_autocast
    with _action_expert_autocast(states):
        ids = self.action_model.generate(states, state=condition, vl_attention_mask=vl_mask,
            temperature=temperature, generator=generator, logits_processor=self.codec.constrain_logits)
    valid = ids != self.codec.pad_id
    with torch.autocast(device_type=self.codec.device.type, enabled=False):
        result = self.codec.decode_actions(ids, examples, valid)
    result["action_token_ids"], result["action_token_mask"] = ids, valid
    result["fast_token_lengths"] = valid.sum(-1)
    if return_history:
        # One snapshot is a completed AR trajectory, not RVQ stage history.
        result["token_history"] = ids[:, None].clone()
    result["action_decode_mode"] = "fast_autoregressive"
    refined = self._apply_discrete_forcing_refinement(result, examples, states, vl_mask, condition,
        continuous_inference_steps=continuous_inference_steps)
    refined["action_token_mask"], refined["fast_token_lengths"] = valid, valid.sum(-1)
    return refined


def contract_fast(self):
    return {"framework": "QwenDiscreteDiffusion", "action_representation": "discrete",
            "codec": self.codec.spec.to_dict(), "action_horizon": self.action_horizon,
            "discrete_action_horizon": self.discrete_action_horizon, "num_layers": self.num_layers,
            "vl_hidden_dim": self.backbone.hidden_size, "backbone_id": self.config.framework.qwenvl.base_vlm,
            "backbone_family": getattr(self.backbone, "family", None),
            "backbone_revision": getattr(self.backbone, "hf_revision", None),
            "fast_autoregressive": self._fast_contract,
            "state_encoding": self._state_encoding,
            "tokenizer": {"frozen": True, "encoder_frozen": True, "codebooks_frozen": True,
                          "decoder_frozen": True, "decoder_adaln_trainable": False}}

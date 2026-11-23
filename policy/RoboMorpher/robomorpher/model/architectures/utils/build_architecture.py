"""Build the QwenDiscreteForcing architecture.

The vision-language backbone is created separately. Each function here sets
one piece: objectives, time projection, conditioning, the discrete head, or
the shared-trunk continuous refiner.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robomorpher.model.backbones.action.continuous_action_head import ContinuousActionHead
from robomorpher.model.backbones.action.discrete_code_head import DiscreteCodeHead
from robomorpher.model.backbones.action.hybrid_action_dit import HybridActionDiT
from robomorpher.model.tokenizer.state_tokenizer import JointStateTokenizer, instruction_state_quantiles, reject_fixed_state_bounds


def _positive_int(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def resolve_execute_horizon(ac, action_horizon):
    """How many predicted steps the robot executes before the next replan."""
    raw = ac.get("execute_horizon", action_horizon)
    value = _positive_int(raw, "action_model.execute_horizon")
    if value > int(action_horizon):
        raise ValueError("action_model.execute_horizon cannot exceed action_horizon")
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def build_objectives(self, cfg, ac, parts):
    self.repeated_diffusion_steps = _positive_int(ac.get("repeated_diffusion_steps", 2), "repeated_diffusion_steps")
    for key, actual in (("vl_hidden_dim", self.backbone.hidden_size), ("num_vl_layers", self.backbone.num_layers)):
        if cfg.framework.qwenvl.get(key) is not None and cfg.framework.qwenvl[key] != actual:
            raise ValueError(f"qwenvl.{key} conflicts with the actual backbone ({actual})")
    self.decode_temperature = float(ac.get("decode_temperature", .1))
    self.choice_temperature = float(ac.get("choice_temperature", .1))
    if any(not math.isfinite(v) or v < 0 for v in (self.decode_temperature, self.choice_temperature)):
        raise ValueError("decode/choice temperatures must be finite and nonnegative")
    # The tokenizer, including decoder AdaLN, stays frozen.
    self.decoder_adaln_trainable = False


def build_horizons(self, cfg, ac, parts):
    self.action_horizon=_positive_int(ac.get("action_horizon",50), "action_horizon")
    self.execute_horizon = resolve_execute_horizon(ac, self.action_horizon)
    self.discrete_action_horizon = _positive_int(
        ac.get("discrete_action_horizon", self.action_horizon),
        "discrete_action_horizon",
    )
    if self.discrete_action_horizon > self.action_horizon:
        raise ValueError("discrete_action_horizon cannot exceed action_horizon")
    self.rvq_stagewise_train = False
    # Official DF uses a fixed linear basis to project the full physical
    # action chunk onto coarse discrete knots, then the same basis to
    # reconstruct the continuous horizon.
    self.discrete_projection_ridge = float(ac.get("discrete_projection_ridge", 1.0e-4))
    if not math.isfinite(self.discrete_projection_ridge) or self.discrete_projection_ridge < 0:
        raise ValueError("discrete_projection_ridge must be finite and nonnegative")
    interpolation = self._build_linear_interpolation_matrix(
        self.discrete_action_horizon, self.action_horizon)
    if self.discrete_action_horizon == self.action_horizon:
        # Avoid shrinking an already matching horizon by 1/(1+ridge).
        # This is the exact identity contract used by official DF.
        projection = torch.eye(self.action_horizon, dtype=interpolation.dtype)
    else:
        gram = interpolation.transpose(0, 1) @ interpolation
        ridge = self.discrete_projection_ridge * torch.eye(
            self.discrete_action_horizon, dtype=interpolation.dtype)
        projection = torch.linalg.solve(gram + ridge, interpolation.transpose(0, 1))
    self.register_buffer("discrete_interpolation_matrix", interpolation, persistent=False)
    self.register_buffer("discrete_projection_matrix", projection, persistent=False)


def build_width_and_conditioning(self, cfg, ac, parts):
    self.num_layers=_positive_int(self.backbone.num_layers if ac.get("num_layers") is None else ac.num_layers, "num_layers")
    if self.num_layers>self.backbone.num_layers:raise ValueError("Action head layers exceed available VLM feature layers")
    hidden=_positive_int(self.backbone.hidden_size if ac.get("hidden_dim") is None else ac.hidden_dim, "hidden_dim")
    # Missing key preserves historical direct K/V projection and checkpoints.
    self.vl_projection = ac.get("vl_projection", "none")
    if self.vl_projection not in ("none", "layernorm_linear"):
        raise ValueError("vl_projection must be none or layernorm_linear")
    cross_dim = hidden if self.vl_projection == "layernorm_linear" else self.backbone.hidden_size
    dit_cfg = dict(ac.get("diffusion_model_cfg", {}))
    dit_cfg.setdefault("interleave_self_attention", True)
    head_dim = _positive_int(dit_cfg.get("attention_head_dim", 64), "diffusion_model_cfg.attention_head_dim")
    heads=_positive_int(max(1,hidden//head_dim) if ac.get("num_heads") is None else ac.num_heads, "num_heads")
    if "attention_head_dim" in dit_cfg and hidden != heads * head_dim:
        raise ValueError("hidden_dim must equal num_heads * diffusion_model_cfg.attention_head_dim")
    self.include_urdf=_boolean(ac.get("urdf_conditioning",True), "urdf_conditioning")
    self.update_urdf_tokens=_boolean(
        ac.get("urdf_token_update", True), "urdf_token_update"
    )
    self.include_state=_boolean(ac.get("state_conditioning",True), "state_conditioning")
    requested_urdf_mode = ac.get("urdf_conditioning_mode", "raw_features")
    if requested_urdf_mode not in {"auto", "raw_features", "tokenizer_encoder", "tokenizer_adaln"}:
        raise ValueError("urdf_conditioning_mode must be auto, raw_features, tokenizer_encoder or tokenizer_adaln")
    feature_dim = int(getattr(self.codec, "urdf_feature_dim", 0))
    self.urdf_mode = ("tokenizer_encoder" if feature_dim else "raw_features") if requested_urdf_mode == "auto" else requested_urdf_mode
    if self.include_urdf and self.urdf_mode in {"tokenizer_encoder", "tokenizer_adaln"} and not feature_dim:
        raise ValueError("This tokenizer has no frozen URDF encoder; choose auto or raw_features")
    if self.urdf_mode == "tokenizer_adaln" and not self.include_urdf:
        raise ValueError("tokenizer_adaln requires action_model.urdf_conditioning=true")
    self.urdf_adaln_conditioning = (
        self.include_urdf and self.urdf_mode == "tokenizer_adaln"
    )
    if self.discrete_mode == "maskgit" and self.include_urdf and not self.urdf_adaln_conditioning:
        raise ValueError("joint_maskgit preserves URDF through tokenizer_adaln, not a separate URDF token stream")
    if "urdf_token_update" in ac and self.urdf_adaln_conditioning:
        raise ValueError(
            "action_model.urdf_token_update has no effect with "
            "urdf_conditioning_mode=tokenizer_adaln because no URDF token stream "
            "exists; delete the key instead of leaving it inert")
    self.use_urdf_encoder = self.include_urdf and self.urdf_mode in {"tokenizer_encoder", "tokenizer_adaln"}
    urdf_feature_dim = feature_dim if self.use_urdf_encoder else 0
    urdf_num_tokens = self.codec.num_arms * self.codec.max_num_joints if self.use_urdf_encoder else 0
    urdf_slot_map = None
    if self.urdf_adaln_conditioning:
        arm_slots = int(self.codec.spec.arm_token_count) + 1
        joint_slots = int(self.codec.latents * self.codec.groups)
        if arm_slots != joint_slots + 1:
            raise ValueError("tokenizer_adaln requires arm-major joint RVQ/PQ slots plus one gripper slot")
        urdf_slot_map = []
        for arm in range(self.codec.num_arms):
            base = arm * self.codec.max_num_joints
            for joint in range(self.codec.max_num_joints):
                urdf_slot_map.extend([[base + joint]] * self.codec.groups)
            urdf_slot_map.append(list(range(base, base + self.codec.max_num_joints)))
        if len(urdf_slot_map) != self.codec.spec.tokens_per_frame:
            raise ValueError("Generated URDF slot map does not match tokenizer token layout")
    # New training recipes opt into text bins. Missing configuration preserves
    # the architecture and checkpoint contract of existing continuous models.
    self._explicit_state_encoding = "state_tokenizer" in ac
    state_cfg = ac.get("state_tokenizer", {
        "mode": "text_bins", "num_bins": 256, "pad_id": 256,
        "text_format": "grouped_numeric", "angle_wrap": "none",
        "angle_low": -2.0 * math.pi, "angle_high": 2.0 * math.pi,
    })
    if not isinstance(state_cfg, Mapping):
        raise ValueError("action_model.state_tokenizer must be a mapping")
    state_cfg = dict(state_cfg)
    explicit_normalization = state_cfg.pop('normalization',None)
    if explicit_normalization is not None and explicit_normalization != 'state_q01_q99':
        raise ValueError('State tokenizer normalization must be state_q01_q99')
    self.state_mode = state_cfg.pop("mode", "text_bins")
    if self.state_mode not in ("text_bins", "continuous"):
        raise ValueError("state_tokenizer.mode must be text_bins or continuous")
    self.state_tokenizer = None
    self._state_encoding = {"mode": self.state_mode, "enabled": self.include_state,
                            "urdf_conditioning": self.include_urdf}
    if self.state_mode == "text_bins":
        reject_fixed_state_bounds(state_cfg)
        unknown = set(state_cfg) - {
            "num_bins", "text_format", "pad_id", "angle_wrap",
            "state_stats", "state_stats_path",
        }
        if unknown:
            raise ValueError(f"Unknown text state settings: {sorted(unknown)}")
        pinned = state_cfg.pop("state_stats", None)
        path = state_cfg.pop("state_stats_path", None)
        if pinned is None and not path:
            data_root = cfg.get("data", {}).get("root")
            if data_root:
                path = str(Path(data_root) / "meta" / "stats_masked.json")
        text_joints = self.physical_action_joints
        state_cfg["state_stats"] = instruction_state_quantiles(
            action_type=self.action_type,
            num_arms=self.codec.num_arms, max_num_joints=text_joints,
            stats_path=path, state_stats=pinned,
            stats_column="action")
        tokenizer = JointStateTokenizer(**state_cfg, normalization="state_q01_q99")
        self._state_encoding["tokenizer"] = tokenizer.metadata()
        self._state_encoding["num_arms"] = self.codec.num_arms
        self._state_encoding["max_num_joints"] = text_joints
        if self.include_state:
            self.state_tokenizer = tokenizer
            if tokenizer.angle_wrap != "none":
                raise ValueError(
                    "The policy state tokenizer must use raw angle_wrap=none")
        if self.include_urdf and not self.use_urdf_encoder and self.codec.state_condition_dim != self.codec.num_arms * (8*self.codec.max_num_joints+1):
            raise ValueError("text_bins URDF conditioning requires the standard joint/URDF codec schema; use continuous for custom state plugins")
    elif set(state_cfg) - {"num_bins", "text_format", "pad_id"}:
        raise ValueError("Unknown continuous state settings")
    # num_bins may remain inherited in continuous mode; it is inactive there,
    # allowing a one-field mode override without changing any state numbers.
    self.head_include_state = self.include_state and self.state_mode == "continuous"
    state_dim=(self.codec.state_condition_dim
               if (self.include_urdf and not self.use_urdf_encoder) or self.head_include_state else 0)
    if ac.get("action_dim") is not None and ac.action_dim != self.codec.num_arms * (self.physical_action_joints + 1):
        raise ValueError("action_dim must match the codec padded physical joint-plus-gripper layout; token slots are codec-derived")
    if ac.get("state_dim") is not None and ac.state_dim != state_dim:
        raise ValueError(f"state_dim conflicts with codec/conditioning-derived dimension {state_dim}")

    parts['hidden'] = hidden
    parts['cross_dim'] = cross_dim
    parts['dit_cfg'] = dit_cfg
    parts['heads'] = heads
    parts['urdf_feature_dim'] = urdf_feature_dim
    parts['urdf_num_tokens'] = urdf_num_tokens
    parts['urdf_slot_map'] = urdf_slot_map
    parts['requested_urdf_mode'] = requested_urdf_mode
    parts['state_dim'] = state_dim


def build_discrete_head(self, cfg, ac, parts):
    hidden = parts['hidden']
    cross_dim = parts['cross_dim']
    dit_cfg = parts['dit_cfg']
    heads = parts['heads']
    urdf_feature_dim = parts['urdf_feature_dim']
    urdf_num_tokens = parts['urdf_num_tokens']
    urdf_slot_map = parts['urdf_slot_map']
    requested_urdf_mode = parts['requested_urdf_mode']
    state_dim = parts['state_dim']
    if "num_inference_steps" in ac:
        inference_steps = _positive_int(ac.num_inference_steps, "num_inference_steps")
    else:
        inference_steps = int(self.discrete_decode_steps)
    future_tokens = _positive_int(ac.get("future_tokens", 32), "future_tokens", minimum=0)
    temporal_cfg = {}
    temporal_enabled = False
    temporal_block_size = int(temporal_cfg.get("block_size", 1))
    temporal_order = str(temporal_cfg.get("rvq_order", "coarse_to_fine"))
    temporal_stage_ids = None
    if temporal_enabled:
        if not getattr(self.codec, "is_frame", False) or int(getattr(self.codec, "groups", 1)) < 2:
            raise ValueError("Temporal RVQ decoding requires the frame tokenizer with at least two RVQ levels")
        gripper_stage = temporal_cfg.get("gripper_stage", None)
        if gripper_stage is None:
            gripper_stage = int(self.codec.groups) - 1
        if isinstance(gripper_stage, bool) or int(gripper_stage) < 0 or int(gripper_stage) >= int(self.codec.groups):
            raise ValueError("action_model.temporal_decode.gripper_stage must be a valid RVQ stage")
        local_stage_ids = [
            stage
            for _latent in range(int(self.codec.latents))
            for stage in range(int(self.codec.groups))
        ] + [int(gripper_stage)]
        temporal_stage_ids = local_stage_ids * int(self.codec.num_arms)
    else:
        gripper_stage = None
    if ac.get("max_seq_len") is not None:
        _positive_int(ac.max_seq_len, "max_seq_len")
    head_mode = "joint_maskgit" if self.discrete_mode == "maskgit" else "categorical"
    self.action_model=DiscreteCodeHead(
        discrete_mode=head_mode,
        vocab_sizes=self.codec.spec.vocab_sizes,action_horizon=self.discrete_action_horizon,
        vl_hidden_dim=cross_dim,num_layers=self.num_layers,
        hidden_dim=hidden,num_heads=heads,state_dim=state_dim,
        urdf_feature_dim=urdf_feature_dim,urdf_num_tokens=urdf_num_tokens,
        update_urdf_tokens=self.update_urdf_tokens,
        urdf_injection=("adaln" if self.urdf_adaln_conditioning else "tokens"),
        urdf_slot_map=urdf_slot_map,
        num_inference_steps=inference_steps,
        num_timestep_buckets=int(ac.get("num_timestep_buckets", 1000)),
        temporal_decode=temporal_enabled,
        temporal_train=bool(temporal_cfg.get("train", True)),
        temporal_train_mode=str(temporal_cfg.get("train_mode", "block_ar")),
        temporal_block_size=temporal_block_size,
        temporal_kv_cache=bool(temporal_cfg.get("kv_cache", True)),
        temporal_rvq_order=temporal_order,
        temporal_stage_ids=temporal_stage_ids,
        # Strict DF has no secondary re-masking. The head keeps the old
        # kwargs only as inert import compatibility; active configs do not
        # expose this feature.
        secondary_remask=bool(ac.get("secondary_remask", False)),
        secondary_remask_threshold=float(ac.get("secondary_remask_threshold", 0.5)),
        df_timestep_semantics=bool(ac.get("discrete_forcing", {}).get("enabled", False)),
        maskgit_enabled=bool(ac.get("maskgit_enabled", False)),
        dropout=float(ac.get("dropout", .2)),future_tokens=future_tokens,
        max_seq_len=ac.get("max_seq_len",None),
        add_pos_embed=ac.get("add_pos_embed", True),
        diffusion_model_cfg=dit_cfg,
    )
    if ac.get("gradient_checkpointing",True):self.action_model.gradient_checkpointing_enable()
    # No policy-owned tokenizer decoder adapters: decoder AdaLN remains in the
    # frozen codec and is never added to optimizer groups.
    self.tokenizer_decoder_adaln = None
    # One independent adapter per consumed VLM layer; self-attention blocks
    # consume no VLM state and must not own unused trainable projectors.
    self.vl_projectors = nn.ModuleDict()
    if self.vl_projection == "layernorm_linear":
        self.vl_projectors.update({str(i): nn.Sequential(
            nn.LayerNorm(self.backbone.hidden_size),
            nn.Linear(self.backbone.hidden_size, hidden),
        ) for i in range(self.num_layers)
            if not (self.action_model.interleave_self_attention and i % 2 == 1)})


    parts['inference_steps'] = inference_steps
    parts['future_tokens'] = future_tokens
    parts['gripper_stage'] = gripper_stage


def build_forcing_refiner(self, cfg, ac, parts):
    hidden = parts['hidden']
    cross_dim = parts['cross_dim']
    dit_cfg = parts['dit_cfg']
    heads = parts['heads']
    urdf_feature_dim = parts['urdf_feature_dim']
    urdf_num_tokens = parts['urdf_num_tokens']
    urdf_slot_map = parts['urdf_slot_map']
    requested_urdf_mode = parts['requested_urdf_mode']
    state_dim = parts['state_dim']
    inference_steps = parts['inference_steps']
    future_tokens = parts['future_tokens']
    gripper_stage = parts['gripper_stage']
    # Discrete Forcing: the categorical action DiT and a reusable continuous
    # refinement head share the first Action-DiT layers and future tokens.
    # RVQ policies train that categorical head with one BD3LM forward.
    df_cfg = ac.get("discrete_forcing", {})
    if df_cfg is None:
        df_cfg = {}
    if not isinstance(df_cfg, Mapping):
        raise ValueError("action_model.discrete_forcing must be a mapping")
    df_unknown = set(df_cfg) - {
        "enabled", "num_shared_layers", "alpha",
        "inference_mode", "flow_loss_weight", "discrete_loss_weight",
        "paired_phase_sampling", "one_step_loss_prob",
        "one_step_loss_weight", "num_discrete_steps", "continuous_loss",
        "continuous_inference_steps", "continuous_stats_path",
    }
    if df_unknown:
        raise ValueError(f"Unknown action_model.discrete_forcing settings: {sorted(df_unknown)}")
    self.discrete_forcing_config = dict(df_cfg)
    self.discrete_forcing_enabled = bool(df_cfg.get("enabled", False))
    self.paired_phase_sampling = False
    self.continuous_refiner = None
    self.register_buffer("_df_q01", torch.empty(0), persistent=False)
    self.register_buffer("_df_q99", torch.empty(0), persistent=False)
    self.discrete_forcing_inference_mode = str(df_cfg.get("inference_mode", "refine"))
    if self.discrete_forcing_inference_mode not in {"discrete", "refine"}:
        raise ValueError("discrete_forcing.inference_mode must be discrete or refine")
    if self.discrete_forcing_enabled:
        if self.action_model.model is None:
            raise ValueError("Discrete Forcing currently requires the categorical_dit action backbone")
        shared_layers = int(df_cfg.get("num_shared_layers", 2))
        if shared_layers < 1 or shared_layers >= self.num_layers:
            raise ValueError("discrete_forcing.num_shared_layers must be between 1 and action_model.num_layers-1")
        # Convert the categorical DiT into one HybridDiT owner. Its shared
        # block objects are used by both branches; the branch blocks and
        # branch timestep/output modules are distinct.
        if not getattr(self.action_model.model, "is_hybrid", False):
            self.action_model.model = HybridActionDiT.from_dit(
                self.action_model.model, shared_layers,
                continuous_output_dim=hidden,
                discrete_output_dim=max(self.codec.spec.vocab_sizes),
                urdf_feature_dim=urdf_feature_dim,
                **({"conditioning_layout": "df_per_token", "future_tokens": future_tokens,
                    "continuous_horizon": self.action_horizon,
                    "max_seq_len": ac.get("max_seq_len", None)}
                   if self.discrete_mode == "maskgit" else {}),
            )
            if self.discrete_mode != "maskgit" or ac.get("gradient_checkpointing", True):
                self.action_model.model.gradient_checkpointing_enable()
        # Metadata alias only; do not register a second module tree under
        # action_model.hybrid_dit because the action_model.model owner is
        # already registered here.
        object.__setattr__(self.action_model, "hybrid_dit", self.action_model.model)
        continuous_layers, continuous_hidden, continuous_heads = self.num_layers, hidden, heads
        discrete_steps = int(df_cfg.get("num_discrete_steps", inference_steps))
        if discrete_steps < 1 or discrete_steps != inference_steps:
            raise ValueError("DF num_discrete_steps must equal action_model.num_inference_steps and be positive")
        continuous_steps = int(df_cfg.get("continuous_inference_steps", 4))
        if continuous_steps < 1:
            raise ValueError("DF continuous_inference_steps must be positive")
        alpha = float(df_cfg.get("alpha", 0.3))
        if not 0.0 < alpha <= 1.0 or not math.isfinite(alpha):
            raise ValueError("discrete_forcing source alpha must be finite and in (0,1]")
        flow_weight = float(df_cfg.get("flow_loss_weight", 1.0))
        # Off unless a config sets both a positive weight and probability.
        # Either zero skips the second continuous DiT forward.
        one_step_weight = float(df_cfg.get("one_step_loss_weight", 0.0))
        discrete_weight = float(df_cfg.get("discrete_loss_weight", 1.0))
        if (flow_weight < 0 or one_step_weight < 0 or discrete_weight < 0
                or not math.isfinite(flow_weight) or not math.isfinite(one_step_weight)
                or not math.isfinite(discrete_weight)):
            raise ValueError("DF loss weights must be finite and nonnegative")
        one_step_prob = float(df_cfg.get("one_step_loss_prob", 0.0))
        if not math.isfinite(one_step_prob) or not 0.0 <= one_step_prob <= 1.0:
            raise ValueError("discrete_forcing.one_step_loss_prob must be finite in [0,1]")
        paired_phase = bool(df_cfg.get("paired_phase_sampling", True))
        if paired_phase:
            phase_repeats = int(self.repeated_diffusion_steps)
            if phase_repeats < 2 or phase_repeats % 2:
                raise ValueError(
                    "discrete_forcing.paired_phase_sampling needs an even "
                    "repeated_diffusion_steps >= 2 so every sample can appear in both phases")
        self.paired_phase_sampling = paired_phase
        self.discrete_forcing_config.update({
            "num_shared_layers": shared_layers,
            "continuous_num_layers": continuous_layers,
            "continuous_hidden_dim": continuous_hidden,
            "continuous_num_heads": continuous_heads,
            "alpha": alpha,
            "branch_input_mode": "full",
            "disjoint_branch_batch_routing": True,
            "isolate_branch_tokens_before_shared": True,
            "allow_branch_communication": False,
            "branch_communication_on_self_attention": False,
            "continuous_source_alpha": alpha,
            "flow_loss_weight": flow_weight,
            "one_step_loss_weight": one_step_weight,
            "discrete_loss_weight": discrete_weight,
            "paired_phase_sampling": paired_phase,
            "one_step_loss_prob": one_step_prob,
            "num_discrete_steps": discrete_steps, "num_continuous_steps": continuous_steps,
            "continuous_inference_steps": continuous_steps,
            "detach_discrete_source": True,
        })
        action_dim = int(self.codec.num_arms * (self.physical_action_joints + 1))
        stats_path = df_cfg.get("continuous_stats_path")
        if not stats_path:
            data_root = cfg.get("data", {}).get("root")
            if data_root:
                stats_path = str(Path(data_root) / "meta" / "stats_masked.json")
        if not stats_path:
            raise ValueError("discrete_forcing requires continuous_stats_path or data.root")
        stats_path = Path(stats_path)
        if not stats_path.is_absolute():
            stats_path = (Path(__file__).resolve().parents[4] / stats_path).resolve()
        if not stats_path.is_file():
            raise FileNotFoundError(f"DF continuous stats not found: {stats_path}")
        with stats_path.open(encoding="utf-8") as handle:
            stats_raw = json.load(handle)
        stats_action = stats_raw.get("action", stats_raw)
        if not isinstance(stats_action, Mapping) or "q01" not in stats_action or "q99" not in stats_action:
            raise ValueError("DF continuous stats must contain action.q01 and action.q99")
        q01 = np.asarray(stats_action["q01"], dtype=np.float32)
        q99 = np.asarray(stats_action["q99"], dtype=np.float32)
        if q01.shape != (action_dim,) or q99.shape != (action_dim,):
            raise ValueError(f"DF q01/q99 must have shape {(action_dim,)}")
        if not np.isfinite(q01).all() or not np.isfinite(q99).all() or np.any(q99 < q01):
            raise ValueError("DF q01/q99 must be finite with q99>=q01")
        self._df_q01 = torch.as_tensor(q01)
        self._df_q99 = torch.as_tensor(q99)
        self.discrete_forcing_config["continuous_stats_path"] = str(stats_path)
        # The checkpoint contract keeps the actual scale. A later edit of the
        # stats file must not change a loaded policy.
        self.discrete_forcing_config["continuous_q01"] = q01.tolist()
        self.discrete_forcing_config["continuous_q99"] = q99.tolist()
        refiner_urdf = int(urdf_feature_dim) if self.urdf_adaln_conditioning else 0
        self.continuous_refiner = ContinuousActionHead(
            discrete_mode=("joint_maskgit" if self.discrete_mode == "maskgit" else "categorical"),
            action_dim=action_dim, action_horizon=self.action_horizon,
            vl_hidden_dim=cross_dim, num_layers=continuous_layers,
            hidden_dim=continuous_hidden, num_heads=continuous_heads,
            state_dim=state_dim, future_tokens=future_tokens,
            max_seq_len=ac.get("max_seq_len", None),
            add_pos_embed=ac.get("add_pos_embed", True),
            dropout=float(ac.get("dropout", .2)), diffusion_model_cfg=dit_cfg,
            loss_type=str(df_cfg.get("continuous_loss", "l2")),
            num_inference_steps=continuous_steps,
            flow_matching_style="starvla",
            noise_beta_alpha=float(ac.get("noise_beta_alpha", 1.5)),
            noise_beta_beta=float(ac.get("noise_beta_beta", 1.0)),
            noise_s=float(ac.get("noise_s", 0.999)),
            num_timestep_buckets=int(ac.get("num_timestep_buckets", 1000)),
            one_step_loss_prob=one_step_prob,
            urdf_feature_dim=refiner_urdf,
            urdf_conditioning=bool(refiner_urdf),
            urdf_arm_count=int(self.codec.num_arms),
            urdf_joints_per_arm=int(self.codec.max_num_joints),
            valid_mask_embedding=bool(ac.get("valid_mask_embedding", True)),
        )
        # One HybridDiT owns shared/continuous/discrete blocks. Do not
        # copy or re-tie individual layers here: both heads point at this
        # single module and branch-specific timestep/output parameters stay
        # independent.
        # Keep one Hybrid owner; the continuous head receives a non-module
        # view so transformer_blocks introspection sees shared +
        # continuous branch without registering duplicate parameters.
        # The Hybrid DiT is owned and registered by action_model. Keep a
        # non-registered continuous branch view on the refiner so its
        # ``transformer_blocks`` inspection exposes shared+continuous
        # blocks without duplicating parameters in the optimizer/state dict.
        self.continuous_refiner._modules.pop("model", None)
        object.__setattr__(
            self.continuous_refiner, "model",
            self.action_model.model.branch_view("continuous"),
        )
        # Keep a single Hybrid parameter owner; this is an introspection
        # alias and must not duplicate state_dict entries.
        object.__setattr__(self.continuous_refiner, "hybrid_dit", self.action_model.model)
        if self.discrete_mode == "maskgit":
            # Shared DF parameters live only under the Hybrid owner. The heads
            # read that owner dynamically, never hold registered aliases.
            self.action_model.future_tokens = None
            self.action_model.position_embedding = None
            self.continuous_refiner.future_tokens = None
            self.continuous_refiner.position_embedding = None
        # Strict DF branch isolation: the continuous sequence contains only
        # physical action tokens (plus state/future tokens). Discrete RVQ IDs
        # are not concatenated here; the decoded physical source is the only
        # discrete-to-continuous conditioning signal.
        # only: retain the branch-local future-token registration and
        # its existing checkpoint keys. New joint models instead use the
        # unique Hybrid owner above, never two registered Parameter aliases.
        if ac.get("gradient_checkpointing", True):
            self.continuous_refiner.gradient_checkpointing_enable()
    self._diffusion_contract = {
        "repeated_diffusion_steps": self.repeated_diffusion_steps,
        "action_horizon": int(self.action_horizon),
        "discrete_action_horizon": int(self.discrete_action_horizon),
        "discrete_projection": {
            "method": "linear_basis_ridge_pseudoinverse",
            "ridge": float(self.discrete_projection_ridge),
            "source_horizon": int(self.action_horizon),
            "target_horizon": int(self.discrete_action_horizon),
        },
        "train_mask_schedule": self.action_model.train_mask_schedule,
        "decode_schedule": self.action_model.decode_schedule,
        "hidden_dim": hidden, "num_heads": heads,
        "future_tokens": (int(self.action_model.model.shared_future_tokens.shape[0])
                          if self.discrete_mode == "maskgit"
                          else int(self.action_model.future_tokens.shape[0])),
        "state_dim": state_dim,
        "urdf_conditioning": {"requested_mode": requested_urdf_mode,
            "resolved_mode": self.urdf_mode if self.include_urdf else "disabled",
            "feature_dim": urdf_feature_dim, "num_tokens": urdf_num_tokens,
            "injection": ("adaln" if self.urdf_adaln_conditioning else "tokens"),
            "slot_map": urdf_slot_map,
            "projector": "per_slot_adaln" if self.urdf_adaln_conditioning else (
                "linear_per_joint" if self.use_urdf_encoder else None),
            "encoder_frozen": bool(self.use_urdf_encoder),
            "token_update": bool(self.update_urdf_tokens)},
        "add_pos_embed": self.action_model.add_pos_embed,
        "position_capacity": (int(self.action_model.model.shared_position_embedding.num_embeddings)
                              if self.discrete_mode == "maskgit" else
                              (int(self.action_model.position_embedding.num_embeddings)
                               if self.action_model.position_embedding is not None else None)),
        "dropout": float(ac.get("dropout", .2)),
        "diffusion_model_cfg": dict(dit_cfg),
        "num_inference_steps": self.action_model.num_inference_steps,
        "num_timestep_buckets": self.action_model.num_timestep_buckets,
        "decode_temperature": self.decode_temperature,
        "choice_temperature": self.choice_temperature,
    }
    # Record the current URDF token-update choice in the contract so a
    # checkpoint cannot silently switch between frozen and mutable tokens.
    if self.use_urdf_encoder and not self.update_urdf_tokens:
        self._diffusion_contract["urdf_conditioning"]["token_update"] = False
    if "temporal_decode" in ac:
        self._diffusion_contract["temporal_decode"] = {
            "enabled": bool(self.action_model.temporal_decode),
            "stagewise_train": bool(self.rvq_stagewise_train),
            "train": bool(self.action_model.temporal_train),
            "train_mode": str(self.action_model.temporal_train_mode),
            "maskgit_enabled": bool(self.action_model.maskgit_enabled),
            "block_size": int(self.action_model.temporal_block_size),
            "kv_cache": bool(self.action_model.temporal_kv_cache),
            "rvq_order": self.action_model.temporal_rvq_order,
            "gripper_stage": (int(gripper_stage) if gripper_stage is not None else None),
            "stage_schedule": (
                "RVQ1_then_RVQ2_plus" if self.action_model.temporal_decode
                else "all_stages_parallel"
            ),
            "attention": (
                "block_causal_v1" if self.action_model.temporal_decode
                else "full_action_parallel"
            ),
            "within_block": "single_pass_parallel",
        }
    if "backbone" in ac:
        self._diffusion_contract["backbone"] = "categorical_dit"
    if self.vl_projection != "none":
        self._diffusion_contract["vl_projection"] = {
            "mode": self.vl_projection, "input_dim": self.backbone.hidden_size,
            "output_dim": hidden, "layer_indices": [int(i) for i in self.vl_projectors],
        }
    if self.discrete_forcing_enabled:
        self._diffusion_contract["discrete_forcing"] = {
            **self.discrete_forcing_config,
            "enabled": True, "inference_mode": self.discrete_forcing_inference_mode,
            "shared_layers": int(self.discrete_forcing_config["num_shared_layers"]),
            "continuous_action_dim": int(self.codec.num_arms * (self.physical_action_joints + 1)),
            "continuous_loss": self.continuous_refiner.loss_type,
            "valid_mask_embedding": bool(self.continuous_refiner.valid_mask_embedding_enabled),
            "urdf_adaln": bool(self.continuous_refiner.urdf_conditioning),
            "refiner_urdf": {
                "feature_dim": int(self.continuous_refiner.urdf_feature_dim),
                "arm_count": int(self.continuous_refiner.urdf_arm_count),
                "joints_per_arm": int(self.continuous_refiner.urdf_joints_per_arm),
                "condition_width": int(self.continuous_refiner.urdf_condition_width),
                "pooling": "per_arm_mean",
            },
            # Stagewise inference performs one categorical forward per
            # RVQ stage, even though each stage itself uses one solver
            # step.  The old parallel DF path has one categorical call.
            "discrete_forward_count": (2 if self.action_model.temporal_decode else int(
                self.discrete_forcing_config["num_discrete_steps"])),
            "nfe": (2 if self.action_model.temporal_decode else int(
                self.discrete_forcing_config["num_discrete_steps"])) + int(
                self.discrete_forcing_config["continuous_inference_steps"]),
            "architecture": "hybrid_dit_shared_trunk_v1",
            "branch_isolation": "disjoint_after_shared",
            "isolate_branch_tokens_before_shared": True,
            "timestep_embeddings": ["global", "continuous", "discrete"],
            "num_timestep_buckets": int(self.continuous_refiner.num_timestep_buckets),
            "continuous_conditioning": "decoded_physical_source_only",
            "continuous_source_train": "gt_token_decode",
            "continuous_source_inference": "predicted_token_decode",
        }
    if self.discrete_mode == "maskgit":
        self._diffusion_contract["discrete_mode"] = "maskgit"
        self._diffusion_contract["maskgit"] = {
            "version": 1, **dict(ac.maskgit),
            "ce_reduction": "mean_over_all_masked_valid_tokens",
            "visible_ce_weight": 0.0,
            "time_formula": "d=1-current_valid_mask_count/valid_count;global=d/2",
            "time_bucket": "floor(t*1000).clamp(0,999)",
            "inference_rng": "independent_discrete_and_continuous_generators",
            "reference_ddvla": "d2efbc467c4669b7918935cab1506afe8cee3662",
            "reference_df": "acce1e9f3ec81fb345a40083f64a0031146f4398",
        }
        self._diffusion_contract["discrete_forcing"].update({
            "architecture": "df_per_token_maskgit_v1",
            "branch_isolation": "disjoint_before_shared",
            "conditioning_layout": "prefix_global_action_local_shared_and_private",
            "action_time_encoder": "sinusoidal_concat_mlp",
            "shared_future_and_position_owner": True,
            "discrete_position_offset": self.action_horizon,
            "output_adaln": False,
            "one_step_subset_execution": "select_before_forward",
        })
    self.codec.eval()

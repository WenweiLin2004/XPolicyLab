"""Assemble a Qwen VLM, a frozen action tokenizer, and a discrete-forcing action DiT.

FAST is the same assembly with an autoregressive discrete head. The continuous
flow policy lives in qwen_pi_v3.py.
"""
from __future__ import annotations
from collections.abc import Mapping
import math
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf
from robomorpher.model.tokenizer.action_codec import build_action_codec
from robomorpher.utils.model.masking import binary_bool_mask

SUPPORTED_MODELS = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
)


def _action_expert_autocast(states):
    """Honor the trainer's AMP precision; frozen codec calls stay explicitly FP32."""
    reference = states[0] if isinstance(states, (list, tuple)) else states
    if reference.device.type == "cuda":
        dtype = (torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda")
                 else reference.dtype)
        if dtype in (torch.bfloat16, torch.float16):
            return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type=reference.device.type, enabled=False)


_DISCRETE_MODES = {"maskgit", "autoregressive"}


def _normalize_discrete_mode(ac, tokenizer_type):
    """MaskGIT for RoboMorpher and 256-bin. FAST is autoregressive."""
    if str(ac.get("action_representation", "")) != "discrete":
        return
    mode = ac.get("discrete_mode", None)
    if mode is None and not tokenizer_type:
        return
    if mode is None:
        mode = "autoregressive" if tokenizer_type == "fast" else "maskgit"
    if mode not in _DISCRETE_MODES:
        raise ValueError("discrete_mode must be maskgit or autoregressive")
    if tokenizer_type == "fast" and mode != "autoregressive":
        raise ValueError("FAST only allows discrete_mode=autoregressive")
    if mode == "autoregressive" and tokenizer_type != "fast":
        raise ValueError("discrete_mode=autoregressive requires action_tokenizer.type=fast")
    ac.discrete_mode = mode
    if mode != "maskgit":
        if "maskgit" in ac:
            raise ValueError("action_model.maskgit requires discrete_mode=maskgit")
        return
    if ac.get("action_representation") != "discrete":
        raise ValueError("maskgit requires a discrete action representation")


def _normalize_joint_maskgit_config(ac, tokenizer_type=""):
    """Apply the maskgit contract. parallel and autoregressive return earlier."""
    _normalize_discrete_mode(ac, tokenizer_type)
    if str(ac.get("action_representation", "")) != "discrete" or ac.get("discrete_mode") != "maskgit":
        return
    if ac.get("maskgit_enabled", False) or ac.get("secondary_remask", False):
        raise ValueError("joint_maskgit uses its own joint decoder; remove maskgit/remask flags")
    if ac.get("backbone", "categorical_dit") != "categorical_dit":
        raise ValueError("joint_maskgit requires the DF categorical DiT backbone")
    if not ac.get("add_pos_embed", True):
        raise ValueError("joint_maskgit requires the shared DF learned position table")
    if ac.get("num_timestep_buckets", 1000) != 1000:
        raise ValueError("joint_maskgit contract uses 1000 timestep buckets")
    raw = ac.get("maskgit", {})
    if not isinstance(raw, Mapping):
        raise ValueError("action_model.maskgit must be a mapping")
    defaults = {
        "decode_steps": 8, "train_layout": "joint_bidirectional",
        "mask_time_distribution": "uniform", "full_mask_prob": 0.0,
        "discrete_time_conditioning": "from_input_mask_ratio",
        "candidate_sampling": "categorical", "choice_temperature": 1.0,
        "remask_committed": False, "action_kv_cache": False,
    }
    unknown = set(raw) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown action_model.maskgit settings: {sorted(unknown)}")
    resolved = {**defaults, **dict(raw)}
    steps = _positive_int(resolved["decode_steps"], "maskgit.decode_steps")
    for key, expected in defaults.items():
        if key == "decode_steps":
            continue
        if resolved[key] != expected:
            raise ValueError(f"joint_maskgit requires maskgit.{key}={expected!r}")
    for key, expected in (("decode_temperature", 1.0), ("choice_temperature", 1.0),
                          ("num_inference_steps", steps)):
        if key in ac and ac[key] != expected:
            raise ValueError(f"action_model.{key} conflicts with the joint_maskgit contract ({expected!r})")
        ac[key] = expected
    ac.maskgit = resolved
    ac.setdefault("repeated_diffusion_steps", 8)
    df = ac.get("discrete_forcing", {})
    if not isinstance(df, Mapping) or "discrete_forcing" not in ac:
        raise ValueError("joint_maskgit requires discrete_forcing")
    if "enabled" in df and not _boolean(df.get("enabled"), "discrete_forcing.enabled"):
        raise ValueError("joint_maskgit requires discrete_forcing.enabled=true")
    ac.discrete_forcing.enabled = True
    if "num_discrete_steps" in df and df["num_discrete_steps"] != steps:
        raise ValueError("discrete_forcing.num_discrete_steps conflicts with maskgit.decode_steps")
    ac.discrete_forcing.num_discrete_steps = steps
    if not df.get("paired_phase_sampling", True):
        raise ValueError("joint_maskgit requires paired_phase_sampling=true")
    dit = dict(ac.get("diffusion_model_cfg", {}))
    for key, expected in (("positional_embeddings", None), ("interleave_self_attention", True),
                          ("norm_type", "ada_norm")):
        if key in dit and dit[key] != expected:
            raise ValueError(f"joint_maskgit requires diffusion_model_cfg.{key}={expected!r}")
        dit[key] = expected
    ac.diffusion_model_cfg = dit


def _normalize_framework_config(config, *, require_action_tokenizer=True):
    """Validate the current policy schema without checkpoint aliases."""
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True)
                           if OmegaConf.is_config(config) else config)
    if "framework" not in cfg:
        raise ValueError("framework configuration is required")
    fw = cfg.framework
    if fw.get("name") not in {"QwenDiscreteForcing", "QwenDiscreteDiffusion", "QwenPI_v3"}:
        raise ValueError("framework.name must be QwenDiscreteForcing or QwenPI_v3")
    unknown = set(fw) - {"name", "qwenvl", "action_model"}
    if unknown:
        raise ValueError(f"Unknown framework settings: {sorted(unknown)}")
    if "qwenvl" not in fw or "action_model" not in fw:
        raise ValueError("framework.qwenvl and framework.action_model are required")
    qw = fw.qwenvl
    qw_known = {"base_vlm", "attn_implementation", "vl_hidden_dim", "num_vl_layers",
                "freeze", "gradient_checkpointing"}
    unknown = set(qw) - qw_known
    if unknown:
        raise ValueError(f"Unknown qwenvl settings: {sorted(unknown)}")
    for key in ("freeze", "gradient_checkpointing"):
        if key in qw:
            _boolean(qw[key], f"qwenvl.{key}")
    ac = fw.action_model
    known = {
        "action_model_type", "action_dim", "state_dim", "action_horizon", "execute_horizon", "discrete_action_horizon", "discrete_projection_ridge",
        "repeated_diffusion_steps", "add_pos_embed", "max_seq_len", "future_tokens",
        "num_inference_steps", "secondary_remask",
        "secondary_remask_threshold",
        "diffusion_model_cfg", "hidden_dim", "num_layers", "num_heads", "dropout",
        "vl_projection", "gradient_checkpointing", "backbone",
        "urdf_conditioning", "urdf_conditioning_mode", "urdf_token_update",
        "state_conditioning", "state_tokenizer", "decode_temperature",
        "choice_temperature", "discrete_forcing",
        "action_representation", "continuous_loss",
        "noise_beta_alpha", "noise_beta_beta", "noise_s",
        "num_timestep_buckets", "num_arms", "max_num_joints", "angle_low",
        "angle_high", "continuous_normalization", "continuous_stats_path",
        "valid_mask_embedding", "value_low", "value_high",
        "maskgit_enabled",
        "fast_autoregressive", "discrete_mode", "maskgit",
    }
    unknown = set(ac) - known
    if unknown:
        raise ValueError(f"Unknown action_model settings: {sorted(unknown)}")
    # New EE methods are explicit opt-ins. Do not insert defaults into saved
    # checkpoint configs, since the normalized framework is part of their contract.
    state = ac.get("state_tokenizer")
    if isinstance(state, Mapping):
        from robomorpher.model.tokenizer.state_tokenizer import reject_fixed_state_bounds
        state_bounds = dict(state)
        if 'normalization' in state_bounds:
            if state_bounds.pop('normalization') != 'state_q01_q99':
                raise ValueError('State tokenizer normalization must be state_q01_q99')
        reject_fixed_state_bounds(state_bounds)
    if ac.get("action_model_type", "LayerwiseDD") != "LayerwiseDD":
        raise ValueError("action_model.action_model_type must be LayerwiseDD")
    for key in ("gradient_checkpointing", "urdf_conditioning", "urdf_token_update", "state_conditioning"):
        if key in ac:
            _boolean(ac[key], f"action_model.{key}")
    if ac.get("backbone", "categorical_dit") != "categorical_dit":
        raise ValueError("action_model.backbone must be categorical_dit")
    if "action_representation" not in ac:
        raise ValueError("action_model.action_representation is required")
    representation = str(ac.action_representation)
    if representation not in {"discrete", "continuous"}:
        raise ValueError("action_model.action_representation must be discrete or continuous")
    if representation == "discrete" and require_action_tokenizer and "action_tokenizer" not in cfg:
        raise ValueError("action_tokenizer is required unless an explicit codec is supplied")
    if representation == "continuous" and "action_tokenizer" in cfg:
        raise ValueError("continuous action representation must not provide action_tokenizer")
    if "future_tokens" not in ac:
        raise ValueError("action_model.future_tokens is required")
    tokenizer_type = str(cfg.action_tokenizer.get("type", "")) if "action_tokenizer" in cfg else ""
    _normalize_joint_maskgit_config(ac, tokenizer_type)
    return cfg
from robomorpher.model.backbones.vlm.qwen_interface import QwenVLBackbone
from robomorpher.model.architectures.utils.build_architecture import (
    _boolean, _positive_int, resolve_execute_horizon,
    build_discrete_head, build_forcing_refiner, build_horizons,
    build_objectives, build_width_and_conditioning,
)

class QwenDiscreteForcing(nn.Module):
    """Qwen VLM plus categorical diffusion action framework."""
    def __init__(self,config,*,codec=None,backbone=None):
        super().__init__()
        cfg=_normalize_framework_config(config, require_action_tokenizer=(codec is None))
        if cfg.framework.name not in {"QwenDiscreteForcing", "QwenDiscreteDiffusion"}:
            raise ValueError("Only framework.name=QwenDiscreteForcing is supported")
        self.config=cfg
        # Official input processor expects this namespace, unrelated data stay raw.
        if "datasets" not in cfg:cfg.datasets={"policy_data":{}}
        elif "policy_data" not in cfg.datasets:
            cfg.datasets.policy_data = {}
        ac=cfg.framework.action_model
        action_type = str(cfg.get("data", {}).get("action_type", "joint")).lower()
        if action_type != "joint":
            raise ValueError("data.action_type must be joint")
        if cfg.get("data", {}).get("joint_auxiliary", False):
            raise ValueError("end-effector action targets are not supported")
        self.joint_auxiliary = False
        self.action_type = action_type
        self.action_representation = str(ac.action_representation)
        if self.action_representation == "continuous":
            raise ValueError("Continuous flow is QwenPI_v3, not discrete forcing")
        if codec is not None:
            self.codec = codec
        else:
            codec_cfg = OmegaConf.to_container(cfg.action_tokenizer, resolve=True)
            codec_cfg.setdefault("action_representation", "joint")
            if codec_cfg.get("type") == "fast" and not codec_cfg.get("stats_path") and codec_cfg.get("q01") is None:
                data_root = cfg.get("data", {}).get("root")
                if not data_root:
                    raise ValueError("FAST stats require data.root or action_tokenizer.stats_path")
                stats_path = Path(data_root) / "meta" / "stats_masked.json"
                if not stats_path.is_absolute():
                    stats_path = (Path(__file__).resolve().parents[3] / stats_path).resolve()
                codec_cfg["stats_path"] = str(stats_path)
            self.codec = build_action_codec(codec_cfg)
        # Learned tokenizers preserve the angle convention stored in their checkpoint.
        # Frozen action targets/decoding are a framework contract, including plugins.
        # eval() alone does not disable gradients through learned condition modules.
        self.codec.requires_grad_(False)
        self.physical_action_joints = self.codec.max_num_joints
        if ac.get("discrete_mode") is None:
            ac.discrete_mode = "autoregressive" if self.codec.spec.kind == "fast" else "maskgit"
        self.discrete_mode = str(ac.discrete_mode)
        if self.discrete_mode == "maskgit":
            if self.codec.spec.kind == "fast":
                raise ValueError("FAST only allows discrete_mode=autoregressive")
            decode = {"mode": "maskgit", "steps": int(ac.maskgit.decode_steps)}
        elif self.discrete_mode == "autoregressive":
            if self.codec.spec.kind != "fast":
                raise ValueError("discrete_mode=autoregressive requires a FAST action tokenizer")
            decode = {"mode": "autoregressive", "steps": int(self.codec.max_tokens)}
        else:
            raise ValueError("discrete_mode must be maskgit or autoregressive")
        self.discrete_decode_mode = decode["mode"]
        self.discrete_decode_steps = int(decode["steps"])
        self.backbone=backbone if backbone is not None else QwenVLBackbone(cfg)
        ac=cfg.framework.action_model
        if ac.get("num_arms") is not None and int(ac.num_arms) != int(self.codec.num_arms):
            raise ValueError("action_model.num_arms does not match the action tokenizer")
        if ac.get("max_num_joints") is not None and int(ac.max_num_joints) != int(self.physical_action_joints):
            raise ValueError("action_model.max_num_joints does not match the action tokenizer")
        fast_config = ac.get("fast_autoregressive", {})
        if not isinstance(fast_config, Mapping):
            raise ValueError("action_model.fast_autoregressive must be a mapping")
        if "enabled" in fast_config and not isinstance(fast_config.get("enabled"), bool):
            raise ValueError("fast_autoregressive.enabled must be boolean")
        # A FAST tokenizer selects this head. enabled has no other legal value.
        fast_enabled = bool(fast_config["enabled"]) if "enabled" in fast_config else self.codec.spec.kind == "fast"
        if self.codec.spec.kind == "fast" or fast_enabled:
            if self.codec.spec.kind != "fast" or not fast_enabled:
                raise ValueError("FAST codec and fast_autoregressive.enabled=true must be selected together")
            from robomorpher.model.backbones.action.fast_refinement import init_fast_refinement
            init_fast_refinement(self, cfg, ac)
            return
        parts = {}
        build_objectives(self, cfg, ac, parts)
        build_horizons(self, cfg, ac, parts)
        build_width_and_conditioning(self, cfg, ac, parts)
        build_discrete_head(self, cfg, ac, parts)
        build_forcing_refiner(self, cfg, ac, parts)

    def train(self,mode=True):
        super().train(mode)
        self.codec.eval()
        return self

    def _backbone_examples(self, examples):
        """Attach observed state to text, without mutating data or reading targets."""
        if self.state_tokenizer is None:
            return examples
        with torch.autocast(device_type=self.state_tokenizer.device.type, enabled=False):
            encoded = self.state_tokenizer.encode_examples(
                examples, num_arms=self.codec.num_arms,
                max_num_joints=self.codec.max_num_joints)
            instructions = self.state_tokenizer.format_instructions(
                [example["lang"] for example in examples], encoded)
        return [dict(example, lang=instruction) for example, instruction in zip(examples, instructions)]

    def _encode_vl_features(self, examples):
        """Project once per observation, before repeat or iterative decoding.

        The original layer indexing is retained. None marks self-attention
        positions, so unused raw VLM tensors are not duplicated by repeat.
        Only the backbone may be frozen; these adapters always remain trainable.
        """
        states, mask = self.backbone(self._backbone_examples(examples), self.num_layers)
        if self.vl_projection == "none":
            return states, mask
        if len(states) != self.num_layers:
            raise ValueError("VLM layer count does not match the projection contract")
        mask = binary_bool_mask(mask, device=states[0].device, name="vl_attention_mask")
        if mask.ndim != 2 or mask.shape[0] != len(examples) or not mask.any(-1).all():
            raise ValueError("VLM mask shape mismatch or entirely padded prompt")
        projected = [None] * self.num_layers
        for index, projector in self.vl_projectors.items():
            h = states[int(index)]
            if h.shape != (*mask.shape, self.backbone.hidden_size):
                raise ValueError("VLM feature shape does not match the projection contract")
            # Sanitize BEFORE LayerNorm/Linear, including backward's saved input.
            h = torch.where(mask[..., None], h, torch.zeros_like(h))
            h = projector(h.to(dtype=projector[1].weight.dtype))
            projected[int(index)] = torch.where(mask[..., None], h, torch.zeros_like(h))
        return projected, mask

    @staticmethod
    def _build_linear_interpolation_matrix(source_horizon, target_horizon):
        source_horizon, target_horizon = int(source_horizon), int(target_horizon)
        if source_horizon < 1 or target_horizon < 1:
            raise ValueError("Interpolation horizons must be positive")
        if source_horizon == target_horizon:
            return torch.eye(source_horizon, dtype=torch.float32)
        positions = torch.linspace(0.0, float(source_horizon - 1), target_horizon)
        left = positions.floor().long()
        right = (left + 1).clamp(max=source_horizon - 1)
        weight = positions - left.to(positions.dtype)
        matrix = torch.zeros((target_horizon, source_horizon), dtype=torch.float32)
        rows = torch.arange(target_horizon)
        matrix[rows, left] += 1.0 - weight
        matrix[rows, right] += weight
        return matrix

    def _project_temporal_array(self, values):
        values = np.asarray(values)
        if values.shape[0] != self.action_horizon:
            raise ValueError("Projection source horizon must equal action_horizon")
        projection = self.discrete_projection_matrix.detach().cpu().numpy()
        flat = values.reshape(values.shape[0], -1).astype(np.float64, copy=False)
        out = projection.astype(np.float64) @ flat
        return out.reshape((self.discrete_action_horizon,) + values.shape[1:]).astype(values.dtype, copy=False)

    def _bounded_uniform_joint_projection(self, examples, projected):
        """Bound derived ridge knots, while retaining strict raw-action checks.

        The 50-to-10 ridge matrix has negative weights, so valid joint chunks
        can produce knots beyond the fixed uniform-bin interval. Only these
        derived targets are saturated; invalid physical input still raises.
        Other codecs and EE resampling retain their original behavior.
        """
        if (self.codec.spec.kind != "uniform_bins"
                or self.codec.action_representation != "joint"):
            return projected
        self.codec.angle_targets(examples)  # Validate original physical labels.
        return np.clip(projected, self.codec.value_low, self.codec.value_high)

    @staticmethod
    def _resample_array(values, target_horizon):
        """Linearly resample a temporal numpy array along axis zero."""
        values = np.asarray(values)
        target_horizon = int(target_horizon)
        if values.shape[0] == target_horizon:
            return values.copy()
        if values.shape[0] < 1 or target_horizon < 1:
            raise ValueError("Temporal arrays need positive source and target horizons")
        if values.shape[0] == 1:
            return np.repeat(values, target_horizon, axis=0)
        src = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float64)
        dst = np.linspace(0.0, 1.0, target_horizon, dtype=np.float64)
        flat = values.reshape(values.shape[0], -1).astype(np.float64, copy=False)
        out = np.stack([np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], axis=1)
        return out.reshape((target_horizon,) + values.shape[1:]).astype(values.dtype, copy=False)

    def _resample_examples(self, examples, target_horizon):
        """Return examples with physical action labels at ``target_horizon``.

        The tokenizer remains frame-local, so coarse discrete labels are made by
        interpolating the physical action chunk before calling ``encode_actions``.
        Validity is nearest-neighbour sampled and therefore never creates a
        partially valid timestep.
        """
        target_horizon = int(target_horizon)
        result = []
        for example in examples:
            # Inference examples contain geometry/state but usually no target
            # action arrays. Decoding only needs the geometry, so leave those
            # mappings untouched while still allowing training resampling.
            if not {"action_angle", "gripper", "action_valid_mask"}.issubset(example):
                result.append(dict(example))
                continue
            q = np.asarray(example["action_angle"])
            g = np.asarray(example["gripper"])
            tv = np.asarray(example["action_valid_mask"], dtype=bool)
            if q.shape[0] != g.shape[0] or q.shape[0] != tv.shape[0]:
                raise ValueError("Action, gripper and validity horizons must agree")
            if q.shape[0] == target_horizon:
                result.append(dict(example))
                continue
            if (q.shape[0] == self.action_horizon and target_horizon == self.discrete_action_horizon
                    and target_horizon != self.action_horizon):
                # Ridge projection is useful for joint-angle trajectories, but
                # it is not range preserving for EE xyz/rot6d either. EE bin
                # targets must remain inside the configured physical bounds,
                # so use the same linear knot interpolation as grippers.
                q2 = (self._resample_array(q, target_horizon)
                      if str(getattr(self.codec, "action_representation", "joint")).lower() == "ee"
                      else self._project_temporal_array(q))
                q2 = self._bounded_uniform_joint_projection([example], q2)
                g2 = self._resample_array(g, target_horizon)
            else:
                q2 = self._resample_array(q, target_horizon)
                g2 = self._resample_array(g, target_horizon)
            src = np.linspace(0.0, 1.0, tv.shape[0], dtype=np.float64)
            dst = np.linspace(0.0, 1.0, target_horizon, dtype=np.float64)
            nearest = np.abs(dst[:, None] - src[None, :]).argmin(axis=1)
            tv2 = tv[nearest]
            result.append(dict(example, action_angle=q2, gripper=g2,
                               action_valid_mask=tv2.astype(bool, copy=False)))
        return result

    def _upsample_decoded_result(self, result, examples):
        """Upsample a coarse token decode to the continuous action horizon."""
        target_horizon = self.action_horizon
        q, g = result["action_angle"], result["gripper"]
        if q.shape[1] == target_horizon:
            return result
        if q.shape[1] != self.discrete_action_horizon:
            raise ValueError("Decoded token horizon does not match discrete_action_horizon")
        b, _, arms, joints = q.shape
        def interp(x):
            # [B,T,...] -> [B,...,H] -> [B,H,...]
            y = x.float().movedim(1, -1).reshape(-1, 1, x.shape[1])
            y = F.interpolate(y, size=target_horizon, mode="linear", align_corners=True)
            return y.reshape(x.movedim(1, -1).shape[:-1] + (target_horizon,)).movedim(-1, 1)
        q2, g2 = interp(q), interp(g)
        # Restoring the fine horizon must also restore its exact time mask.
        # Geometry-only validity would incorrectly unmask repeated short-episode tails.
        time_valid = (self.codec._stack(examples, "action_valid_mask", torch.bool)
                      if all("action_valid_mask" in example for example in examples) else None)
        valid2 = self.codec.valid_tokens(examples, target_horizon, time_valid)
        out = self.codec._result(q2, g2, result["joint_valid_mask"], valid2)
        for key in ("direct_ee",):
            if key in result:
                out[key] = interp(result[key])
        return out

    def _condition(self,examples):
        # In text mode, only URDF stays in the continuous head prefix. Observed
        # angles/grippers have already entered Qwen and must not be injected twice.
        if not(self.head_include_state or self.include_urdf):return None
        with torch.autocast(device_type=self.codec.device.type,enabled=False):
            if self.use_urdf_encoder:
                features, mask = self.codec.encode_urdf_condition(examples)
                continuous = (self.codec.state_condition(examples, include_urdf=False, include_state=True)
                              if self.head_include_state else None)
                return {"urdf_features": features, "urdf_valid_mask": mask, "state": continuous}
            return self.codec.state_condition(examples,
                include_urdf=self.include_urdf,include_state=self.head_include_state)

    def _df_stats(self, device, dtype):
        if self._df_q01.numel() == 0 or self._df_q99.numel() == 0:
            raise RuntimeError("DF fine-tune q01/q99 stats are not initialized")
        return (self._df_q01.to(device=device, dtype=dtype),
                self._df_q99.to(device=device, dtype=dtype))

    def _normalize_df_physical(self, physical, mask):
        physical = physical.masked_fill(~mask, 0)
        lo, hi = self._df_stats(physical.device, physical.dtype)
        # Some single-embodiment datasets have an arm that stays stationary.
        # A zero-width quantile interval encodes at the center and decodes to lo.
        constant = hi == lo
        width = torch.where(constant, torch.ones_like(hi), hi - lo)
        values = 2.0 * (physical - lo) / width - 1.0
        return values.clamp(-2.2, 2.2).masked_fill(~mask | constant, 0)

    def _df_refiner_condition(self, state):
        """Validate/select conditioning passed to the DF continuous branch."""
        if self.continuous_refiner is None:
            raise RuntimeError("Discrete Forcing refiner is not initialized")
        needs_urdf = bool(self.continuous_refiner.urdf_conditioning)
        if needs_urdf:
            if not isinstance(state, Mapping):
                raise ValueError(
                    "DF continuous refiner requires a mapping with "
                    "urdf_features and urdf_valid_mask")
            for key in ("urdf_features", "urdf_valid_mask"):
                if not torch.is_tensor(state.get(key)):
                    raise ValueError(f"DF refiner condition is missing tensor {key}")
        # Plain state tensors (or an optional state field in a mapping) remain
        # normal state inputs; URDF only goes to a branch with AdaLN modules.
        return state if needs_urdf else None

    def _denormalize_df(self, normalized):
        lo, hi = self._df_stats(normalized.device, normalized.dtype)
        return 0.5 * (normalized + 1.0) * (hi - lo) + lo

    def _pack_continuous_action_result(self, result, examples):
        """Pack hard decoded actions into the DF fine-tune normalized layout."""
        q = result["action_angle"]
        g = result["gripper"]
        token_valid = binary_bool_mask(result["token_valid_mask"], device=q.device, name="token_valid_mask")
        b, h, _, _ = q.shape
        arm_tokens = token_valid.reshape(b, h, self.codec.num_arms, -1)
        joint_valid = arm_tokens[..., :-1].any(-1)[..., None] & result["joint_valid_mask"][:, None]
        gripper_valid = arm_tokens[..., -1]
        physical = torch.cat([q, g[..., None]], dim=-1).flatten(2)
        mask = torch.cat([joint_valid, gripper_valid[..., None]], dim=-1).flatten(2)
        return self._normalize_df_physical(physical, mask), mask

    def _continuous_targets_from_examples(self, examples):
        with torch.autocast(device_type=self.codec.device.type, enabled=False):
            _, m, _, g, tv = self.codec._action_inputs(examples)
            q, q_valid = self.codec.angle_targets(examples)
            g_valid = tv[:, :, None] & m.any(-1)[:, None]
            physical = torch.cat([q, g[..., None]], dim=-1).flatten(2)
            mask = torch.cat([q_valid, g_valid[..., None]], dim=-1).flatten(2)
            return self._normalize_df_physical(physical, mask), mask

    def _df_phase_timestep(self, batch, device, *, continuous):
        """Sample the official DF absolute phase time for one branch.

        The Beta draw is interpreted as a noise level, like the reference
        continuous head.  Discrete rows use the complementary half after the
        +.5 shift, while continuous rows use the unshifted half.  Thus the
        resulting global phase is in [0, .5) for discrete and [.5, 1] for
        continuous, with the strict discrete upper bound required by the
        categorical head.
        """
        if self.continuous_refiner is None:
            raise RuntimeError("DF phase sampling requires a continuous refiner")
        sample = self.continuous_refiner.beta_dist.sample((int(batch),)).to(
            device=device, dtype=torch.float32
        ) * 0.5
        if not continuous:
            sample = sample + 0.5
        noise_s = float(self.continuous_refiner.noise_s)
        phase = (noise_s - sample) / noise_s
        return phase.clamp(0.5 if continuous else 0.0,
                           1.0 if continuous else 0.5 - 1.0e-6)

    def _discrete_forcing_loss(self, states, vl_mask, state, ids, valid, examples,
                               *, continuous_examples=None, phase_timestep=None):
        if self.continuous_refiner is None:
            raise RuntimeError("Discrete Forcing refiner is not initialized")
        if continuous_examples is None:
            continuous_examples = examples
        with torch.no_grad(), torch.autocast(device_type=self.codec.device.type, enabled=False):
            decoded = self.codec.decode_actions(ids, examples, valid)
            decoded = self._upsample_decoded_result(decoded, continuous_examples)
            source, source_valid = self._pack_continuous_action_result(decoded, continuous_examples)
            target, target_valid = self._continuous_targets_from_examples(continuous_examples)
        if source.shape != target.shape or not torch.equal(source_valid, target_valid):
            raise ValueError("DF decoded source and continuous target masks do not match")
        refiner_urdf = self._df_refiner_condition(state)
        with _action_expert_autocast(states):
            return self.continuous_refiner.refinement_loss(
                states, source, target, source_valid, state=state,
                vl_attention_mask=vl_mask, urdf_condition=refiner_urdf,
                alpha=float(self.discrete_forcing_config["alpha"]),
                one_step_loss_weight=float(self.discrete_forcing_config["one_step_loss_weight"]),
                phase_timestep=phase_timestep,
                one_step_loss_prob=float(self.discrete_forcing_config["one_step_loss_prob"]),
            )

    def _repeat_batch(self, states, vl_mask, state, ids, valid, examples, repeats):
        """Block-wise repeat: [sample0..sampleN, sample0..sampleN, ...]."""
        repeats = int(repeats)
        if repeats <= 1:
            return states, vl_mask, state, ids, valid, examples
        if isinstance(states, (list, tuple)):
            states = [None if h is None else h.repeat(repeats, 1, 1) for h in states]
        else:
            states = states.repeat(repeats, *([1] * (states.ndim - 1)))
        ids, valid = ids.repeat(repeats, 1, 1), valid.repeat(repeats, 1, 1)
        vl_mask = vl_mask.repeat(repeats, 1)
        if state is not None:
            if isinstance(state, Mapping):
                state = {k: (None if v is None else v.repeat(repeats, *([1] * (v.ndim - 1))))
                         for k, v in state.items()}
            else:
                state = state.repeat(repeats, *([1] * (state.ndim - 1)))
        return states, vl_mask, state, ids, valid, list(examples) * repeats

    def _paired_phase_forward(self, states, vl_mask, state, ids, valid, examples, continuous_examples=None):
        """Discrete Forcing phase split: each sample feeds each branch separately.

        Every sample is repeated ``repeated_diffusion_steps`` times and half of
        those copies train only the categorical (MaskGIT CE) branch while the
        other half train only the continuous refinement branch, so a single row
        never carries both objectives. Losses are reduced inside their own phase
        and summed before one backward pass, mirroring Discrete Forcing's
        ``paired_phase_sampling`` + ``disjoint_branch_batch_routing`` training.
        """
        repeats = int(self.repeated_diffusion_steps)
        half = max(1, repeats // 2)
        if continuous_examples is None:
            continuous_examples = examples
        d_states, d_mask, d_state, d_ids, d_valid, d_examples = self._repeat_batch(
            states, vl_mask, state, ids, valid, examples, half)
        if self.discrete_mode == "maskgit":
            # The head derives both clocks from its sampled, actual input MASKs.
            with _action_expert_autocast(d_states):
                out = self.action_model(
                    d_states, d_ids, d_valid, state=d_state, vl_attention_mask=d_mask)
            ce = out["action_loss"]
        else:
            # Discrete rows use only the first half of the global diffusion interval.
            d_phase = torch.zeros(d_ids.shape[0], device=d_ids.device, dtype=torch.float32)
            with _action_expert_autocast(d_states):
                out = self.action_model(
                    d_states, d_ids, d_valid, state=d_state, vl_attention_mask=d_mask,
                    phase_timestep=d_phase,
                )
            ce = out["action_loss"]
        discrete_weight = float(self.discrete_forcing_config.get("discrete_loss_weight", 1.0))
        optimized = discrete_weight * ce
        metrics = {"ce_loss": ce.detach()}
        if self.discrete_mode == "maskgit":
            metrics["ce_loss_weighted"] = optimized.detach()
        c_states, c_mask, c_state, c_ids, c_valid, c_token_examples = self._repeat_batch(
            states, vl_mask, state, ids, valid, examples, half)
        # Token IDs are coarse (discrete_action_horizon), while the continuous
        # DF target must stay on the execution horizon (action_horizon). Keep
        # these lists separate: passing c_token_examples as the continuous
        # target silently trains FM on the coarse 10-step target.
        c_continuous_examples = list(continuous_examples) * half
        c_phase = self._df_phase_timestep(c_ids.shape[0], c_ids.device, continuous=True)
        df = self._discrete_forcing_loss(
            c_states, c_mask, c_state, c_ids, c_valid, c_token_examples,
            continuous_examples=c_continuous_examples,
            phase_timestep=c_phase,
        )
        df_weighted = float(self.discrete_forcing_config["flow_loss_weight"]) * df["continuous_loss"]
        optimized = optimized + df_weighted
        metrics.update({
            "continuous_flow_loss": df["continuous_flow_loss"].detach(),
            "continuous_one_step_loss": df["continuous_one_step_loss"].detach(),
            "continuous_loss": df["continuous_loss"].detach(),
            "continuous_loss_weighted": df_weighted.detach(),
        })
        metrics["action_loss"] = optimized
        return metrics

    def forward(self,examples):
        if getattr(self, "_fast_enabled", False):
            from robomorpher.model.backbones.action.fast_refinement import forward_fast
            return forward_fast(self, examples)
        for example in examples:
            if "action_angle" in example and int(np.asarray(example["action_angle"]).shape[0]) != self.action_horizon:
                raise ValueError("Dataset action_chunk_size must equal action_model.action_horizon")
        discrete_examples = self._resample_examples(examples, self.discrete_action_horizon)
        with torch.autocast(device_type=self.codec.device.type,enabled=False):
            ids,valid=self.codec.encode_actions(discrete_examples)
        if ids.shape[1]!=self.discrete_action_horizon:
            raise ValueError("Encoded discrete horizon does not match discrete_action_horizon")
        states,vl_mask=self._encode_vl_features(examples)
        state = self._condition(examples)
        if self.paired_phase_sampling:
            return self._paired_phase_forward(states, vl_mask, state, ids, valid,
                                              list(discrete_examples), list(examples))
        repeats = self.repeated_diffusion_steps
        # VLM and frozen target encoder run ONCE. Tensor.repeat and list * R both
        # produce [sample0, sample1, ..., sample0, sample1, ...], never interleave.
        states, vl_mask, state, ids, valid, discrete_examples = self._repeat_batch(
            states, vl_mask, state, ids, valid, discrete_examples, repeats)
        continuous_examples = list(examples) * repeats
        with _action_expert_autocast(states):
            out=self.action_model(states,ids,valid,state=state,vl_attention_mask=vl_mask)
        ce = out["action_loss"]
        metrics = {"ce_loss": ce.detach()}
        # Discrete Forcing optimizes continuous_loss + w_discrete * discrete_loss
        # (their RobotWin run uses w_discrete = 0.1). The weight is only read
        # when the DF branch is active, so tokenizer-only runs keep 1.0.
        discrete_weight = (float(self.discrete_forcing_config.get("discrete_loss_weight", 1.0))
                           if self.discrete_forcing_enabled else 1.0)
        optimized = discrete_weight * ce
        if self.discrete_forcing_enabled:
            df = self._discrete_forcing_loss(
                states, vl_mask, state, ids, valid, discrete_examples,
                continuous_examples=continuous_examples,
            )
            df_weighted = float(self.discrete_forcing_config["flow_loss_weight"]) * df["continuous_loss"]
            optimized = optimized + df_weighted
            metrics.update({
                "continuous_flow_loss": df["continuous_flow_loss"].detach(),
                "continuous_one_step_loss": df["continuous_one_step_loss"].detach(),
                "continuous_loss": df["continuous_loss"].detach(),
                "continuous_loss_weighted": df_weighted.detach(),
            })
        metrics["action_loss"] = optimized
        return metrics

    def _prediction_result(self, predicted, examples, valid):
        if isinstance(predicted, tuple):
            ids, history = predicted
        else:
            ids, history = predicted, None
        decode_examples = self._resample_examples(examples, self.discrete_action_horizon)
        with torch.autocast(device_type=self.codec.device.type,enabled=False):
            result = self.codec.decode_actions(ids, decode_examples, valid)
        result = self._upsample_decoded_result(result, examples)
        result["action_token_ids"] = ids
        if history is not None:
            result["token_history"] = history
        # normalized action API is meaningful only for the uniform-bin
        # baseline. Learned tokenizers retain their explicit physical-unit API.
        if self.codec.spec.kind == "uniform_bins":
            q = 2 * (result["action_angle"] - self.codec.angle_low) / (self.codec.angle_high-self.codec.angle_low) - 1
            g = result["gripper"] * 2 - 1
            mask = result["joint_valid_mask"]
            packed = []
            for i in range(len(examples)):
                arms = []
                for arm in range(self.codec.num_arms):
                    if bool(mask[i, arm].any()):
                        arms.extend([q[i, :, arm, mask[i, arm]], g[i, :, arm, None]])
                packed.append(torch.cat(arms, -1).float().cpu().numpy())
            result["normalized_actions"] = np.stack(packed) if len({x.shape for x in packed}) == 1 else packed
        return result

    @staticmethod
    def _resolve_joint_generator(generator, device):
        """Create an inference-only stream without consuming the other stream.

        Default seeds use CPU global RNG, so restoring its state reproduces
        both streams independently of CUDA sampling consumption. Supplied
        generators retain their current state; initial_seed alone does not
        replay a generator that a caller has already advanced.
        """
        device = torch.device(device)
        if generator is None:
            seed = int(torch.randint(0, torch.iinfo(torch.int64).max, (),
                                     device="cpu", dtype=torch.int64).item())
            return torch.Generator(device=device).manual_seed(seed)
        if not isinstance(generator, torch.Generator):
            raise TypeError("joint inference generators must be torch.Generator instances")
        if torch.device(generator.device) != device:
            raise ValueError(f"Inference generator device {generator.device} does not match action device {device}")
        return generator

    def _joint_inference_generators(self, generator=None, continuous_generator=None):
        """Shared full/split-server entry point for independent D and FM RNGs."""
        if self.discrete_mode != "maskgit":
            raise ValueError("Independent inference streams are versioned by joint_maskgit")
        if generator is not None and generator is continuous_generator:
            raise ValueError("Discrete and continuous inference require independent generators")
        device = next(self.action_model.parameters()).device
        return (self._resolve_joint_generator(generator, device),
                self._resolve_joint_generator(continuous_generator, device))

    @torch.inference_mode()
    def _apply_discrete_forcing_refinement(self, result, examples, states, vl_mask, state,
                                           *, continuous_inference_steps=None,
                                           continuous_generator=None):
        if not self.discrete_forcing_enabled or self.discrete_forcing_inference_mode != "refine":
            return result
        steps = (int(self.discrete_forcing_config["continuous_inference_steps"])
                 if continuous_inference_steps is None else int(continuous_inference_steps))
        if steps < 1:
            raise ValueError("continuous_inference_steps must be positive")
        source, source_valid = self._pack_continuous_action_result(result, examples)
        compute_dtype = states[0].dtype if isinstance(states, (list, tuple)) else states.dtype
        source = source.to(dtype=compute_dtype)
        if self.discrete_mode == "maskgit":
            continuous_generator = self._resolve_joint_generator(continuous_generator, source.device)
            noise = torch.randn(source.shape, device=source.device, dtype=source.dtype,
                                generator=continuous_generator).masked_fill(~source_valid, 0)
        else:
            noise = torch.randn_like(source).masked_fill(~source_valid, 0)
        alpha = float(self.discrete_forcing_config["alpha"])
        source = (alpha * source + (1.0 - alpha) * noise).masked_fill(~source_valid, 0)
        refiner_urdf = self._df_refiner_condition(state)
        with _action_expert_autocast(states):
            refined = self.continuous_refiner.predict_from_source(
                states, source, source_valid, state=state, vl_attention_mask=vl_mask,
                urdf_condition=refiner_urdf,
                num_inference_steps=steps)
        b, h, _, j = result["action_angle"].shape
        # DF continuous targets and the refiner sampler use the same q01/q99
        # normalization, whose valid clipped range is [-2.2, 2.2].  Clipping
        # here to [-1, 1] silently saturated large joint commands and made
        # decode/refine inference disagree with training.
        normalized = refined.clamp(-2.2, 2.2)
        physical = self._denormalize_df(normalized)
        # Inverse of _pack_continuous_action_result: each arm's gripper
        # immediately follows that arm's joints, not both grippers at the end.
        shaped = physical.reshape(b, h, self.codec.num_arms, self.physical_action_joints + 1)
        shaped_valid = source_valid.reshape_as(shaped)
        q, g = shaped[..., :-1], shaped[..., -1]
        q = q.masked_fill(~shaped_valid[..., :-1], 0)
        g = g.clamp(0.0, 1.0).masked_fill(~shaped_valid[..., -1], 0)
        refined_result = self.codec._result(q, g, result["joint_valid_mask"], result["token_valid_mask"])
        refined_result["action_angle_discrete"] = result["action_angle"]
        refined_result["packed_actions_discrete"] = result["packed_actions"]
        refined_result["action_token_ids"] = result["action_token_ids"]
        if "token_history" in result:
            refined_result["token_history"] = result["token_history"]
        if "discrete_decode_config" in result:
            # Codec._result rebuilds the physical result; retain the explicit
            # joint solver override instead of losing it at the FM boundary.
            refined_result["discrete_decode_config"] = dict(result["discrete_decode_config"])
            refined_result["continuous_decode_config"] = {"num_inference_steps": steps}
            if self.discrete_mode == "maskgit":
                refined_result["continuous_decode_config"]["generator_seed"] = continuous_generator.initial_seed()
        refined_result["action_decode_mode"] = "discrete_forcing_refine"
        refined_result["discrete_forcing_source"] = source
        return refined_result

    @torch.inference_mode()
    def predict_action(self,examples,*,decode_temperature=None,choice_temperature=None,
                       num_inference_steps=None,generator=None,return_history=False,
                       secondary_remask=None, secondary_remask_threshold=None,
                       continuous_inference_steps=None,
                       continuous_generator=None):
        """Predict hard discrete tokens, optionally followed by DF refinement."""
        if self.discrete_mode == "maskgit":
            generator, continuous_generator = self._joint_inference_generators(generator, continuous_generator)
        elif continuous_generator is not None:
            raise ValueError("continuous_generator is supported only for joint_maskgit")
        if getattr(self, "_fast_enabled", False):
            from robomorpher.model.backbones.action.fast_refinement import predict_fast
            return predict_fast(self, examples, decode_temperature=decode_temperature,
                                generator=generator, return_history=return_history,
                                continuous_inference_steps=continuous_inference_steps)
        if not isinstance(examples,list):examples=[examples]
        if self.discrete_forcing_enabled:
            if self.discrete_mode == "maskgit":
                if secondary_remask:
                    raise ValueError("joint_maskgit does not remask committed tokens")
                if num_inference_steps is not None:
                    num_inference_steps = _positive_int(num_inference_steps, "num_inference_steps")
                else:
                    num_inference_steps = int(self.discrete_forcing_config["num_discrete_steps"])
            else:
                # Keep the historical fixed categorical count for old models.
                num_inference_steps = int(self.discrete_forcing_config["num_discrete_steps"])
            secondary_remask = False
        valid=self.codec.valid_tokens(examples,self.discrete_action_horizon)
        states,vl_mask=self._encode_vl_features(examples)
        condition = self._condition(examples)
        with _action_expert_autocast(states):
            ids=self.action_model.predict(states,valid,state=condition,
                vl_attention_mask=vl_mask,
                decode_temperature=self.decode_temperature if decode_temperature is None else decode_temperature,
                choice_temperature=self.choice_temperature if choice_temperature is None else choice_temperature,
                num_inference_steps=num_inference_steps,
                generator=generator, return_history=return_history,
                secondary_remask=secondary_remask,
                secondary_remask_threshold=secondary_remask_threshold)
        result = self._prediction_result(ids, examples, valid)
        if self.discrete_mode == "maskgit":
            result["discrete_decode_config"] = {
                "mode": "maskgit", "num_inference_steps": int(num_inference_steps),
                "decode_temperature": self.decode_temperature if decode_temperature is None else float(decode_temperature),
                "choice_temperature": self.choice_temperature if choice_temperature is None else float(choice_temperature),
                "time_conditioning": "from_input_mask_ratio",
                "generator_seed": generator.initial_seed(),
            }
        return self._apply_discrete_forcing_refinement(
            result, examples, states, vl_mask, condition,
            continuous_inference_steps=continuous_inference_steps,
            continuous_generator=continuous_generator)

    def checkpoint_contract(self):
        if getattr(self, "_fast_enabled", False):
            from robomorpher.model.backbones.action.fast_refinement import contract_fast
            return contract_fast(self)
        contract = {"framework": self.config.framework.name, "codec":self.codec.spec.to_dict(),
                "action_horizon":self.action_horizon,
                "discrete_action_horizon":self.discrete_action_horizon,
                "num_layers":self.num_layers,
                "vl_hidden_dim":self.backbone.hidden_size,
                "backbone_id":self.config.framework.qwenvl.base_vlm,
                "backbone_family":getattr(self.backbone,"family",None),
                "backbone_revision":getattr(self.backbone,"hf_revision",None)}
        contract["discrete_diffusion"] = self._diffusion_contract
        contract["tokenizer"] = {
            "frozen": True,
            "encoder_frozen": True,
            "codebooks_frozen": True,
            "decoder_frozen": True,
            "decoder_adaln_trainable": False,
        }
        if self._explicit_state_encoding:
            # Training resume must reject changes to state binning, layout or route.
            contract["state_encoding"] = self._state_encoding
        return contract

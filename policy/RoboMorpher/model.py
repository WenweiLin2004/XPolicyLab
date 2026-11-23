"""RoboTwin/XPolicyLab policy entry point.

The adapter constructs and owns the trained RoboMorpher network in this module.
Checkpoint, tokenizer, stats and diffusion contracts are verified before
inference. deploy.py contains only the simulator action loop.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

POLICY_ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = POLICY_ROOT / "checkpoints"
if str(POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICY_ROOT))


def _import_xpolicylab():
    """Eval imports XPolicyLab. Training does not look for a bench."""
    try:
        from XPolicyLab.model_template import ModelTemplate
        from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
        return ModelTemplate, candidate_checkpoint_roots
    except ImportError:
        root = os.environ.get("XPOLICYLAB_ROOT")
        if not root:
            return None, None
        parent = str(Path(root).expanduser().resolve().parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from XPolicyLab.model_template import ModelTemplate
        from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
        return ModelTemplate, candidate_checkpoint_roots


from robomorpher.utils.inference_paths import bind_checkpoint_assets, dataset_asset_id, select_checkpoint

if __package__:
    from .robot_metadata import resolve_robot_bundle
else:
    from robot_metadata import resolve_robot_bundle


ModelTemplate, candidate_checkpoint_roots = _import_xpolicylab()
if ModelTemplate is None:
    class ModelTemplate:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Start eval from <bench>/XPolicyLab/policy/RoboMorpher")

CAMERA_MAP = {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}
_CAMERA_TO_XPL = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}



def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _host_neutral(item):
    """Compare project paths after the host prefix, keeping the project suffix."""
    # Both layouts refer to the same released dataset. Numeric stats and
    # fingerprints elsewhere in the contract still undergo their usual checks.
    dataset = dataset_asset_id(item)
    if dataset:
        return dataset
    for marker in ("EmbodiBench/", "RoboTwin/"):
        index = item.find(marker)
        if index >= 0:
            return item[index:]
    return item


def _contract_for_compare(value):
    """Canonicalize stored project paths without weakening the saved contract.

    A checkpoint may record a host-specific prefix in front of EmbodiBench or
    RoboTwin. The comparison keeps that project suffix and the rest of the
    contract unchanged.
    """
    value = _canonical(value)
    def replace(item):
        if isinstance(item, str):
            item = _host_neutral(item)
            # Both names describe the same QwenPI-v3 action encoder: the
            # sinusoidal embedding is the flow-time input to the action MLP.
            if item == "qwenpi_v3_action_mlp_with_sinusoidal":
                item = "qwenpi_v3_action_mlp_with_sinusoidal_time"
            return item
        if isinstance(item, list):
            return [replace(x) for x in item]
        if isinstance(item, dict):
            return {k: replace(v) for k, v in item.items()}
        return item
    return replace(value)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _codec_spec_matches(saved, runtime):
    """Compare codec contracts while allowing the known continuous stats mount alias.

    Continuous codec fingerprints include the stats path for provenance. A saved
    path and the runtime path can use different host prefixes. The full model contract below still validates the actual
    continuous q01/q99 values and all model settings; discrete RoboMorpher fingerprints
    remain byte-exact.
    """
    if _canonical(saved) == _canonical(runtime):
        return True
    if not isinstance(saved, dict) or not isinstance(runtime, dict):
        return False
    if saved.get("kind") != "continuous" or runtime.get("kind") != "continuous":
        return False
    left = dict(saved)
    right = dict(runtime)
    left.pop("fingerprint", None)
    right.pop("fingerprint", None)
    return _contract_for_compare(left) == _contract_for_compare(right)


def _df_stats_view(saved, runtime):
    """Old DF checkpoints recorded only the stats path.

    New checkpoints also store continuous_q01/q99. Those values are installed
    before this comparison, so a replaced stats file cannot change the scale.
    A checkpoint saved before the values existed still loads against its file.
    """
    left = _contract_for_compare(saved)
    right = _contract_for_compare(runtime)
    saved_forcing = ((left.get("discrete_diffusion") or {}).get("discrete_forcing")
                     if isinstance(left, dict) else None)
    runtime_diffusion = right.get("discrete_diffusion") if isinstance(right, dict) else None
    runtime_forcing = (runtime_diffusion.get("discrete_forcing")
                       if isinstance(runtime_diffusion, dict) else None)
    if (isinstance(saved_forcing, dict) and isinstance(runtime_forcing, dict)
            and "continuous_q01" not in saved_forcing):
        runtime_forcing.pop("continuous_q01", None)
        runtime_forcing.pop("continuous_q99", None)
    for contract in (left, right):
        if not isinstance(contract, dict):
            continue
        contract.pop("training_mode", None)
        contract.pop("angle_aux_loss", None)
        flow = contract.get("continuous_flow")
        if isinstance(flow, dict):
            flow.pop("continuous_flow_matching", None)
            norm = flow.get("normalization")
            if isinstance(norm, dict):
                if norm.get("mode") == "slot_q99_q01":
                    norm["mode"] = "q99"
                norm.pop("physical_bounds", None)
        diffusion = contract.get("discrete_diffusion")
        if isinstance(diffusion, dict):
            # Earlier MaskGIT contracts repeated both schedules in this subsection.
            maskgit = diffusion.get("maskgit")
            if isinstance(maskgit, dict):
                for key in ("train_mask_schedule", "decode_schedule"):
                    if key in maskgit and maskgit[key] == diffusion.get(key):
                        del maskgit[key]
            diffusion.pop("no_mask_token_prob", None)
            diffusion.pop("use_simple_max", None)
            diffusion.pop("rtc", None)
            forcing = diffusion.get("discrete_forcing")
            if isinstance(forcing, dict):
                forcing.pop("continuous_inference_mode", None)
            if diffusion.get("representation") == "bin":
                diffusion.pop("representation")
    return left, right


def _model_contract_matches(saved, runtime):
    """Strict model contract comparison with continuous path-only aliases.

    The continuous codec spec already has its semantic fields checked separately,
    while its fingerprint includes a host-local stats path.  Keep that one hash
    out of this second comparison; q01/q99 and stats path are still compared in
    continuous_flow.normalization below.
    """
    left, right = _df_stats_view(saved, runtime)
    # Pre-EE QwenPI-v3 checkpoints implicitly used joint actions. Accept only
    # those exact historical defaults; explicit EE/different schemas stay strict.
    if (isinstance(left, dict) and isinstance(right, dict)
            and left.get("framework") == right.get("framework") == "QwenPI_v3"):
        saved_flow, runtime_flow = left.get("continuous_flow"), right.get("continuous_flow")
        if (isinstance(saved_flow, dict) and isinstance(runtime_flow, dict)
                and "action_type" not in saved_flow and "ee_pose" not in saved_flow
                and runtime_flow.get("action_type") == "joint"
                and runtime_flow.get("ee_pose", "__missing__") is None):
            saved_flow["action_type"] = "joint"
            saved_flow["ee_pose"] = None
    if (
        isinstance(left, dict) and isinstance(right, dict)
        and left.get("action_representation") == "continuous"
        and right.get("action_representation") == "continuous"
        and isinstance(left.get("codec"), dict)
        and isinstance(right.get("codec"), dict)
    ):
        left["codec"] = dict(left["codec"])
        right["codec"] = dict(right["codec"])
        left["codec"]["fingerprint"] = "__continuous_codec_verified__"
        right["codec"]["fingerprint"] = "__continuous_codec_verified__"
    return left == right


def resolve_checkpoint(value=None):
    return select_checkpoint(POLICY_ROOT, value)


def _runtime_device(value):
    """Resolve CUDA aliases after CUDA_VISIBLE_DEVICES has been applied."""
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def model_storage_bytes(model, precision="bf16", prefixes=None):
    """Deployment storage estimate; deduplicates shared/tied tensors."""
    size = 0
    seen = set()
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if prefixes is not None and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        key = id(value)
        if key in seen:
            continue
        seen.add(key)
        element = value.element_size()
        if value.is_floating_point():
            element = 4 if name.startswith(("codec.", "state_tokenizer.")) or precision == "fp32" else 2
        size += value.numel() * element
    return size


def load_inference_model(checkpoint=None, *, device=None):
    """Construct/load on CPU from a model state_dict. Optimizer is not stored."""
    from robomorpher.utils.training.checkpointing import load_weight_checkpoint
    from robomorpher.model.policy_factory import build_policy
    path = resolve_checkpoint(checkpoint)
    weights, payload = load_weight_checkpoint(path)
    required = {"config", "codec_spec", "contract", "global_step"}
    if not required <= payload.keys():
        raise ValueError(f"Not a complete policy checkpoint: missing {required - payload.keys()}")
    config = OmegaConf.create(payload["config"])
    saved_backbone = config.framework.qwenvl.base_vlm
    bind_checkpoint_assets(POLICY_ROOT, config, path)
    if device is not None and torch.device(device).type == "cpu":
        if config.framework.qwenvl.get("attn_implementation") == "flash_attention_2":
            config.framework.qwenvl.attn_implementation = "sdpa"
    saved_contract = payload["contract"].get("model_contract", payload["contract"])
    _translate_saved_continuous_config(config)
    _install_pinned_state_quantiles(config, saved_contract)
    with torch.device("cpu"):
        model = build_policy(config)
    saved_contract = payload["contract"].get("model_contract", payload["contract"])
    _install_pinned_df_stats(model, saved_contract)
    if not _codec_spec_matches(payload["codec_spec"], model.codec.spec.to_dict()):
        raise ValueError("Action tokenizer spec/fingerprint does not match policy checkpoint")
    runtime_contract = model.checkpoint_contract()
    # Compare the serialized backbone ID after opening its relocated local files.
    if runtime_contract.get("backbone_id") == config.framework.qwenvl.base_vlm:
        runtime_contract["backbone_id"] = saved_backbone
    if not _model_contract_matches(saved_contract, runtime_contract):
        raise ValueError("Policy backbone/config/diffusion/state contract does not match checkpoint")
    manifest_path = next((candidate / "checkpoint_manifest.json"
                          for candidate in (path.parent, path.parent.parent)
                          if (candidate / "checkpoint_manifest.json").is_file()), None)
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text())
        encoded = json.dumps(_canonical(payload["contract"]), sort_keys=True).encode()
        if manifest.get("contract_sha256") != hashlib.sha256(encoded).hexdigest():
            raise ValueError("Checkpoint contract differs from directory manifest")
    state = dict(weights)
    if any(name.startswith(("codec.", "state_tokenizer.", "action_tokenizer.")) for name in state):
        raise ValueError("Native policy checkpoint unexpectedly includes codec tensors")
    # In continuous mode this state_dict is an empty physical-layout object;
    # in discrete mode it binds the frozen tokenizer tensors and statistics.
    # Frozen codec tensors are bound by its SHA256, architecture, and exact stats.
    state.update({f"codec.{name}": tensor for name, tensor in model.codec.state_dict().items()})
    live = model.state_dict()
    for name, value in list(state.items()):
        current = live.get(name)
        if torch.is_tensor(value) and torch.is_tensor(current) and value.dtype != current.dtype:
            state[name] = value.to(dtype=current.dtype)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Incomplete inference state: {result}")
    model.requires_grad_(False)
    model.eval()
    metadata = {"checkpoint": str(path), "global_step": int(payload["global_step"]),
                "codec_spec": _canonical(payload["codec_spec"]),
                "contract": _canonical(payload["contract"]),
                "model_storage_bf16_bytes": model_storage_bytes(model, "bf16")}
    del state, payload
    gc.collect()
    return model, config, metadata


def _split_head_modules(model):
    """Modules that run with the action head, including the continuous refiner.

    FAST, MaskGIT, BD3LM and 256-bin all keep ``continuous_refiner`` as its own
    module. Only the shared DiT lives under ``action_model``. Leaving the
    refiner on the VLM device breaks split-GPU inference.
    """
    names = {"codec", "action_model"}
    if getattr(model, "continuous_refiner", None) is not None:
        names.add("continuous_refiner")
    return names


def _saved_df_stats(saved_contract):
    """q01/q99 pinned in a DF checkpoint, if this checkpoint recorded them."""
    if not isinstance(saved_contract, dict):
        return None
    diffusion = saved_contract.get("discrete_diffusion")
    forcing = diffusion.get("discrete_forcing") if isinstance(diffusion, dict) else None
    if not isinstance(forcing, dict) or "continuous_q01" not in forcing:
        return None
    return forcing["continuous_q01"], forcing["continuous_q99"]


def _translate_saved_continuous_config(config):
    """Drop removed continuous switches and rename the saved normalization mode."""
    action = config.framework.action_model
    action.pop("continuous_flow_matching", None)
    action.pop("ee_bounds_mode", None)
    action.pop("rtc", None)
    forcing = action.get("discrete_forcing")
    if isinstance(forcing, Mapping):
        mode = str(forcing.pop("continuous_inference_mode", "")).lower()
        mirrored_steps = forcing.pop("num_continuous_steps", None)
        if "continuous_inference_steps" not in forcing and mirrored_steps is not None:
            forcing["continuous_inference_steps"] = mirrored_steps
        source_alpha = forcing.pop("continuous_source_alpha", None)
        if source_alpha is not None:
            forcing["alpha"] = source_alpha
        if mode == "one_step":
            forcing["continuous_inference_steps"] = 1
        for key in ("continuous_num_layers", "continuous_hidden_dim", "continuous_num_heads",
                    "detach_discrete_source", "branch_input_mode", "conditioning_layout",
                    "disjoint_branch_batch_routing", "isolate_branch_tokens_before_shared",
                    "allow_branch_communication", "branch_communication_on_self_attention"):
            forcing.pop(key, None)
    name = action.get("continuous_normalization")
    if name == "slot_q99_q01":
        action.continuous_normalization = "q99"
    elif name == "physical_bounds":
        action.continuous_normalization = "minmax"


def _install_pinned_state_quantiles(config, saved_contract):
    """Rebuild instruction-text bins from the quantiles recorded in the checkpoint."""
    if not isinstance(saved_contract, dict):
        return
    encoding = saved_contract.get("state_encoding")
    tokenizer = encoding.get("tokenizer") if isinstance(encoding, dict) else None
    pinned = tokenizer.get("normalization") if isinstance(tokenizer, dict) else None
    if not isinstance(pinned, dict) or "q01" not in pinned:
        return
    action = config.framework.action_model
    state = action.setdefault("state_tokenizer", {})
    for key in ("angle_low", "angle_high", "normalization", "value_low", "value_high"):
        state.pop(key, None)
    state["state_stats"] = pinned


def _install_pinned_df_stats(model, saved_contract):
    """Use the checkpoint's FM scale, not whatever the stats file says now."""
    pinned = _saved_df_stats(saved_contract)
    if pinned is None or not hasattr(model, "_df_q01"):
        return
    q01, q99 = pinned
    model._df_q01 = torch.tensor(q01, dtype=torch.float32)
    model._df_q99 = torch.tensor(q99, dtype=torch.float32)
    forcing = getattr(model, "_diffusion_contract", {}).get("discrete_forcing")
    if isinstance(forcing, dict):
        forcing["continuous_q01"] = list(q01)
        forcing["continuous_q99"] = list(q99)
    live = getattr(model, "discrete_forcing_config", None)
    if isinstance(live, dict):
        live["continuous_q01"] = list(q01)
        live["continuous_q99"] = list(q99)


def deploy_inference_model(model, options):
    device = torch.device(options.get("device") or "cpu")
    head_device = torch.device(options.get("head_device") or device)
    if head_device != device and (head_device.type != "cuda" or device.type != "cuda"):
        raise ValueError("Split deployment requires explicit CUDA backbone and head devices")
    precision = str(options.get("precision", "bf16" if device.type == "cuda" else "fp32"))
    if precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    split = head_device != device
    head_modules = _split_head_modules(model)
    buckets = [(device, "memory_budget_gib", 16.0,
                ("backbone.", "vl_projectors.", "state_tokenizer.") if split else None)]
    if split:
        buckets.append((head_device, "head_memory_budget_gib", 8.0, tuple(f"{name}." for name in sorted(head_modules))))
    deployment = []
    for target, budget_key, default_budget, prefixes in buckets:
        if target.type != "cuda":
            continue
        budget_gib = float(options.get(budget_key, default_budget))
        reserve_gib = float(options.get("reserve_free_gib", 2.0))
        if not math.isfinite(budget_gib) or budget_gib <= 0 or not math.isfinite(reserve_gib) or reserve_gib < 0:
            raise ValueError("Invalid CUDA memory budget/reserve")
        free, total = torch.cuda.mem_get_info(target)
        budget = int(budget_gib * 2**30)
        storage = model_storage_bytes(model, precision, prefixes)
        if free < budget + reserve_gib * 2**30:
            raise RuntimeError(f"Insufficient safe free memory on {target}: free={free/2**30:.3f} GiB, budget={budget_gib:.3f}, reserved={reserve_gib:.3f}")
        if storage + 512 * 2**20 >= budget:
            raise RuntimeError(f"Model weights on {target} need {storage/2**30:.3f} GiB; budget {budget_gib:.3f} leaves insufficient activation room")
        torch.cuda.set_per_process_memory_fraction(min(budget / total, 1.0), target)
        deployment.append({"device": str(target), "memory_budget_bytes": budget,
                           "model_storage_bytes": storage, "gpu_free_before_bytes": free,
                           "reserve_free_bytes": int(reserve_gib * 2**30)})
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    for name, child in model.named_children():
        target = head_device if name in head_modules else device
        child.to(device=target) if name in {"codec", "state_tokenizer"} else child.to(device=target, dtype=dtype)
    model.requires_grad_(False).eval()
    model.codec.eval()
    return {"device": str(device), "head_device": str(head_device),
            "precision": precision, "deployment_devices": deployment}


def _none_like(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"})


def _image(value: Any, name: str) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] < 3 or not array.size:
        raise ValueError(f"{name} must be an RGB image, got {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmax(array)) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")



def _apply_df_inference_override(model, usr_args):
    """Apply deploy-time Discrete Forcing solver settings to a loaded model.

    The deployment YAML is shared by every policy, so solver settings must be
    harmless for checkpoints that have no DF branch: in that case they are
    ignored (with an explicit note) instead of failing the load. Returns the
    metadata entry describing the applied override, or ``None``.
    """
    steps = usr_args.get("continuous_inference_steps")
    discrete_mode = usr_args.get("discrete_inference_mode")
    if discrete_mode is None:
        discrete_mode = os.environ.get("ROBOMORPHER_DISCRETE_INFERENCE_MODE")
    if steps is None and discrete_mode is None:
        return None
    if not getattr(model, "discrete_forcing_enabled", False):
        print("[RoboMorpher] DF inference override ignored: checkpoint has no "
              "Discrete Forcing branch", flush=True)
        return None
    saved = model.discrete_forcing_config
    resolved_discrete_mode = str(
        discrete_mode if discrete_mode is not None
        else getattr(model, "discrete_forcing_inference_mode", saved.get("inference_mode", "refine"))
    ).lower()
    if resolved_discrete_mode not in {"discrete", "refine"}:
        raise ValueError("discrete_inference_mode must be discrete or refine")
    resolved_steps = int(steps if steps is not None else saved["continuous_inference_steps"])
    if resolved_steps < 1:
        raise ValueError("continuous_inference_steps must be positive")
    model.discrete_forcing_inference_mode = resolved_discrete_mode
    model.discrete_forcing_config = dict(
        saved, inference_mode=resolved_discrete_mode,
        continuous_inference_steps=resolved_steps, num_continuous_steps=resolved_steps)
    return dict(steps=resolved_steps, discrete_mode=resolved_discrete_mode,
                checkpoint_steps=saved["continuous_inference_steps"],
                checkpoint_discrete_mode=saved.get("inference_mode", "refine"))

class _PolicyRuntime:
    def __init__(self, usr_args, *, model=None, config=None, metadata=None):
        self.usr_args = dict(usr_args)
        if model is None:
            model, config, metadata = load_inference_model(
                usr_args.get("checkpoint") or usr_args.get("vla_model_path"),
                device=usr_args.get("device") or "cpu",
            )
        self.model = model
        self.config = config if OmegaConf.is_config(config) else OmegaConf.create(config)
        self.metadata = dict(metadata or {})
        self.device = _runtime_device(usr_args.get("device") or "cpu")
        self.head_device = _runtime_device(usr_args.get("head_device") or self.device)
        if self.head_device != self.device and (self.head_device.type != "cuda" or self.device.type != "cuda"):
            raise ValueError("Split deployment requires explicit CUDA backbone and head devices")
        self.precision = str(usr_args.get("precision", "bf16" if self.device.type == "cuda" else "fp32"))
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be bf16 or fp32")
        self.execution_horizon = int(getattr(self.model, "execute_horizon", self.model.action_horizon))
        if not 1 <= self.execution_horizon <= self.model.action_horizon:
            raise ValueError("action_model.execute_horizon must be 1..action_horizon")
        # MaskGIT honors an explicit num_inference_steps from the eval entry.
        # BD3LM, 256-bin and FAST keep the count stored with the checkpoint.
        self._df_enabled = bool(getattr(self.model, "discrete_forcing_enabled", False))
        if self._df_enabled:
            self.discrete_decode_mode = str(getattr(self.model, "discrete_decode_mode", "fixed"))
            saved_steps = getattr(self.model, "discrete_decode_steps", None)
            if saved_steps is None:
                saved_steps = getattr(self.model, "discrete_forcing_config", {}).get("num_discrete_steps", 1)
            requested = self.usr_args.get("num_inference_steps")
            if requested is not None and self.discrete_decode_mode == "maskgit":
                if isinstance(requested, bool) or int(requested) != requested or int(requested) < 1:
                    raise ValueError("num_inference_steps must be a positive integer")
                self.num_inference_steps = int(requested)
            else:
                self.num_inference_steps = int(saved_steps)
        elif getattr(self.model, "action_representation", None) == "continuous":
            # The shared deploy YAML controls a pure continuous checkpoint
            # through continuous_inference_steps (normally four).
            self.num_inference_steps = int(usr_args.get(
                "continuous_inference_steps",
                self.config.framework.action_model.get("num_inference_steps", 4),
            ))
        else:
            self.num_inference_steps = int(usr_args.get(
                "num_inference_steps",
                self.config.framework.action_model.get("num_inference_steps", 8),
            ))
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        self.secondary_remask = bool(usr_args.get(
            "secondary_remask",
            self.config.framework.action_model.get("secondary_remask", False),
        ))
        self.secondary_remask_threshold = float(usr_args.get(
            "secondary_remask_threshold",
            self.config.framework.action_model.get("secondary_remask_threshold", 0.5),
        ))
        if not 0.0 <= self.secondary_remask_threshold <= 1.0:
            raise ValueError("secondary_remask_threshold must be in [0,1]")
        # Inference overrides are applied only after strict checkpoint loading.
        # They do not modify checkpoint files, training settings, or loss weights.
        override = _apply_df_inference_override(self.model, usr_args)
        if override is not None:
            self.metadata["df_inference_override"] = override
            print(f"[RoboMorpher] DF inference override applied | "
                  f"continuous_steps={override['steps']} "
                  f"refinement={override['discrete_mode']}", flush=True)
        if self._df_enabled:
            # Re-read after a deploy-time DF override. Secondary remasking is
            # a MaskGIT feature and is forbidden in strict DF.
            df = self.model.discrete_forcing_config
            self.continuous_inference_steps = int(df["continuous_inference_steps"])
            self.secondary_remask = False
            if self.discrete_decode_mode == "autoregressive":
                discrete_note = f"discrete_decode=autoregressive max_tokens={self.num_inference_steps}"
            else:
                discrete_note = f"discrete_steps={self.num_inference_steps}"
            print(f"[RoboMorpher] inference | {discrete_note} "
                  f"refinement={self.model.discrete_forcing_inference_mode} "
                  f"continuous_steps={self.continuous_inference_steps}", flush=True)
        else:
            self.continuous_inference_steps = None
        self.camera_names = tuple(self.config.data.camera_names)
        if any(name not in CAMERA_MAP for name in self.camera_names):
            raise ValueError(f"Unsupported training camera_names {self.camera_names}")
        self.max_joints = int(self.model.codec.max_num_joints)
        if int(self.model.codec.num_arms) != 2:
            raise ValueError("This EmbodiBench adapter currently expects a dual-arm codec")
        self.reset()
        self._deploy()

    def _deploy(self):
        split = self.head_device != self.device
        head_modules = _split_head_modules(self.model)
        buckets = [(self.device, "memory_budget_gib", 6.0,
                    ("backbone.", "vl_projectors.", "state_tokenizer.") if split else None)]
        if split:
            buckets.append((self.head_device, "head_memory_budget_gib", 3.0, tuple(f"{name}." for name in sorted(head_modules))))
        deployment = []
        for device, budget_key, default_budget, prefixes in buckets:
            if device.type != "cuda":
                continue
            budget_gib = float(self.usr_args.get(budget_key, default_budget))
            reserve_gib = float(self.usr_args.get("reserve_free_gib", 1.25))
            if not math.isfinite(budget_gib) or budget_gib <= 0 or not math.isfinite(reserve_gib) or reserve_gib < 0:
                raise ValueError("Invalid CUDA memory budget/reserve")
            free, total = torch.cuda.mem_get_info(device)
            budget = int(budget_gib * 2**30)
            storage = model_storage_bytes(self.model, self.precision, prefixes)
            if free < budget + reserve_gib * 2**30:
                raise RuntimeError(f"Insufficient safe free memory on {device}: free={free/2**30:.3f} GiB, "
                                   f"budget={budget_gib:.3f}, reserved_for_training={reserve_gib:.3f}")
            if storage + 512 * 2**20 >= budget:
                raise RuntimeError(f"Model weights on {device} need {storage/2**30:.3f} GiB; "
                                   f"budget {budget_gib:.3f} leaves insufficient activation room")
            torch.cuda.set_per_process_memory_fraction(min(budget / total, 1.0), device)
            deployment.append({"device": str(device), "memory_budget_bytes": budget,
                               "model_storage_bytes": storage, "gpu_free_before_bytes": free,
                               "reserve_free_bytes": int(reserve_gib * 2**30)})
        # No model.bfloat16(): codec and state bins must preserve FP32 boundaries.
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float32
        for name, child in self.model.named_children():
            target = self.head_device if name in head_modules else self.device
            child.to(device=target) if name in {"codec", "state_tokenizer"} else child.to(device=target, dtype=dtype)
        self.model.requires_grad_(False).eval()
        self.model.codec.eval()
        self.metadata.update(device=str(self.device), head_device=str(self.head_device),
                             precision=self.precision, num_inference_steps=self.num_inference_steps,
                             secondary_remask=self.secondary_remask,
                             secondary_remask_threshold=self.secondary_remask_threshold,
                             deployment_devices=deployment)

    def reset(self):
        self.robot_signature = None
        self.robot_context = None

    def close(self):
        """Release this policy process's model; never touch training processes."""
        self.reset()
        if self.model is not None:
            self.model.to("cpu")
            self.model = None
        gc.collect()
        for device in {self.device, self.head_device}:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    def update_robot_context(self, paths, joint_names):
        if len(paths) != 2 or len(joint_names) != 2:
            raise ValueError("Exactly two ordered arm URDFs/joint name lists required")
        paths = [Path(path).resolve(strict=True) for path in paths]
        signature = tuple((str(path), _sha256(path), tuple(names)) for path, names in zip(paths, joint_names))
        if signature == self.robot_signature:
            return
        from robomorpher.utils.data.urdf_features import prepare_tokenizer_chain as prepare_chain
        features = np.zeros((2, self.max_joints, 6), dtype=np.float32)
        mask = np.zeros((2, self.max_joints), dtype=bool)
        refs = []
        for arm, (path, names) in enumerate(zip(paths, joint_names)):
            names = list(names)
            if len(set(names)) != len(names) or not 0 <= len(names) <= self.max_joints:
                raise ValueError(f"Invalid arm {arm} joint name order/count: {names}")
            # An absent arm has no valid joints, URDF features, or gripper state.
            # This is the same zero padding used by the training dataset.
            if not names:
                refs.append({"path": str(path), "joint_names": [], "absent": True})
                continue
            actual = prepare_chain(path, names, embed_arm_base=False)
            features[arm, :len(names)] = actual["features"]
            mask[arm, :len(names)] = True
            refs.append({"path": str(path), "sha256": signature[arm][1],
                         "joint_names": names})
        if not mask.any():
            raise ValueError("At least one observed arm must have valid joints")
        self.robot_context = {"urdf_features": features, "joint_valid_mask": mask,
                              "arm_valid_mask": mask.any(-1),
                              "urdf_paths": [str(path) for path in paths],
                              "joint_names": [list(names) for names in joint_names]}
        self.robot_signature = signature
        self.metadata["robot_context"] = refs
        print(f"[RoboMorpher] episode URDF context: {refs}", flush=True)

    @torch.inference_mode()
    def _predict_routed(self, examples):
        """Run the action head on ``head_device`` for split deployment."""
        if getattr(self.model, "_fast_enabled", False):
            from robomorpher.model.backbones.action.fast_refinement import predict_fast
            states, vl_mask = self.model._encode_vl_features(examples)
            states = [state.to(self.head_device) if state is not None else None for state in states]
            condition = self.model._condition(examples)
            if torch.is_tensor(condition):
                condition = condition.to(self.head_device)
            return predict_fast(
                self.model, examples, states=states, vl_mask=vl_mask.to(self.head_device),
                condition=condition, decode_temperature=self.model.decode_temperature,
                continuous_inference_steps=self.continuous_inference_steps,
            )
        # The categorical branch works on D token frames while the
        # continuous branch works on H physical frames.
        token_horizon = (self.model.discrete_action_horizon
                         if self.model.action_representation != "continuous"
                         else self.model.action_horizon)
        valid = self.model.codec.valid_tokens(examples, token_horizon)
        states, vl_mask = self.model._encode_vl_features(examples)
        states = [state.to(self.head_device) if state is not None else None for state in states]
        vl_mask = vl_mask.to(self.head_device)
        if self.model.action_representation == "continuous":
            state = self.model._condition(examples)
            if isinstance(state, Mapping):
                state = {k: (v.to(self.head_device) if torch.is_tensor(v) else v)
                         for k, v in state.items()}
            elif torch.is_tensor(state):
                state = state.to(self.head_device)
            normalized = self.model.action_model.predict(
                states, valid.to(self.head_device), state=state,
                vl_attention_mask=vl_mask,
                num_inference_steps=self.num_inference_steps)
            return self.model.codec.decode_continuous(
                normalized, examples, valid.to(self.head_device))
        condition = self.model._condition(examples)
        if isinstance(condition, Mapping):
            condition = {k: (v.to(self.head_device) if torch.is_tensor(v) else v)
                         for k, v in condition.items()}
        elif torch.is_tensor(condition):
            condition = condition.to(self.head_device)
        joint_maskgit = getattr(self.model, "discrete_mode", "parallel") == "maskgit"
        discrete_generator = continuous_generator = None
        if joint_maskgit:
            discrete_generator, continuous_generator = self.model._joint_inference_generators()
        ids = self.model.action_model.predict(
            states, valid, state=condition, vl_attention_mask=vl_mask,
            decode_temperature=self.model.decode_temperature,
            choice_temperature=self.model.choice_temperature,
            num_inference_steps=self.num_inference_steps,
            generator=discrete_generator, return_history=False,
            secondary_remask=False if self._df_enabled else self.secondary_remask,
            secondary_remask_threshold=self.secondary_remask_threshold)
        result = self.model._prediction_result(ids, examples, valid)
        if joint_maskgit:
            result["discrete_decode_config"] = {
                "mode": "maskgit", "num_inference_steps": self.num_inference_steps,
                "generator_seed": discrete_generator.initial_seed(),
            }
        if self._df_enabled:
            result = self.model._apply_discrete_forcing_refinement(
                result, examples, states, vl_mask, condition,
                continuous_inference_steps=self.continuous_inference_steps,
                **({"continuous_generator": continuous_generator} if joint_maskgit else {}))
        return result

    def _restore_execution_branches(self, result, example):
        """Validate raw decoded actions without angle post-processing."""
        raw = result["action_angle"]
        mask = np.asarray(example["joint_valid_mask"], dtype=bool)
        expected = (1, self.model.action_horizon, 2, self.max_joints)
        if tuple(raw.shape) != expected:
            raise ValueError(f"Unexpected decoded action_angle shape: {tuple(raw.shape)}")
        decoded_mask = result.get("joint_valid_mask")
        if decoded_mask is None:
            raise ValueError("Decoded result does not contain joint_valid_mask")
        if not np.array_equal(np.asarray(decoded_mask[0], dtype=bool), mask):
            raise ValueError("Decoded and observed joint masks differ")
        packed = result["packed_actions"][0]
        if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(packed).all()):
            raise ValueError("Decoded action contains non-finite values")
        result["decoded_action_angle"] = raw
        result["decoded_packed_actions"] = result["packed_actions"]
        if "normalized_actions" in result:
            result["decoded_normalized_actions"] = result.pop("normalized_actions")
        return result


class Model(_PolicyRuntime, ModelTemplate):
    """Single XPolicyLab adapter for the RoboMorpher policy."""
    def __init__(self, model_cfg):
        ModelTemplate.__init__(self)
        self.model_cfg = dict(model_cfg)
        self.env_cfg_type = str(self.model_cfg.get("env_cfg_type") or "").strip()
        if not self.env_cfg_type:
            raise ValueError("RoboMorpher requires env_cfg_type")
        requested_action_type = str(self.model_cfg.get("action_type") or "joint").lower()
        if requested_action_type not in {"joint", "qpos"}:
            raise ValueError("RoboMorpher action_type must be joint or qpos")
        self.action_type = "joint"
        configured = self.model_cfg.get("robotwin_root") or os.environ.get("ROBOTWIN_ROOT")
        if not configured:
            raise FileNotFoundError("Eval requires the bench root. Start from <bench>/XPolicyLab/policy/RoboMorpher")
        self.robotwin_root = Path(configured).expanduser().resolve()
        if not self.robotwin_root.is_dir():
            raise FileNotFoundError(f"Bench root does not exist: {self.robotwin_root}")
        checkpoint = self._resolve_checkpoint()
        args = dict(self.model_cfg)
        # Keep deploy-time continuous solver overrides for _PolicyRuntime.
        # Execution horizon still comes exclusively from the checkpoint.
        for key in ("replan_steps", "execution_horizon", "execute_horizon"):
            args.pop(key, None)
        args["checkpoint"] = str(checkpoint)
        _PolicyRuntime.__init__(self, args)
        loaded_action_type = str(getattr(self.model, "action_type", "joint")).lower()
        if loaded_action_type != self.action_type:
            raise ValueError(
                f"Deploy action_type={self.action_type} does not match checkpoint action_type={loaded_action_type}"
            )
        self._batch = {}
        self._order = []
        self.robot_signature = None
        self.robot_context = None
        bench = self.model_cfg.get("bench_name") or self.model_cfg.get("dataset_name")
        if not bench:
            # Older launchers only pass the simulator root.
            bench = "EmbodiBench" if (self.robotwin_root / "EmbodiBench").is_dir() else self.robotwin_root.name
        self._dynamic_urdf_required = bench == "EmbodiBench"
        if self._dynamic_urdf_required:
            # Generated EmbodiBench robots must come from each episode's
            # observation, never a static real-robot bootstrap configuration.
            self._joint_names = [[], []]
        else:
            paths, self._joint_names = resolve_robot_bundle(self.robotwin_root, self.env_cfg_type)
            self.update_robot_context(paths, self._joint_names)
        self.default_instruction = str(self.model_cfg.get("default_instruction") or "follow the instruction")
        print(f"[RoboMorpher] ready | env_cfg_type={self.env_cfg_type} joints={[len(x) for x in self._joint_names]} chunk={self.model.action_horizon} execute={self.execution_horizon}")

    def _resolve_checkpoint(self) -> Path:
        cfg = dict(self.model_cfg)
        explicit_keys = ("checkpoint_path", "checkpoint", "model_path", "ckpt_path", "model_dir", "pretrained_path")
        explicit = next((cfg[key] for key in explicit_keys if not _none_like(cfg.get(key))), None)
        value = explicit if explicit is not None else cfg.get("ckpt_name")
        def named_candidates(name):
            # RoboMorpher shares one trained checkpoint across all embodiments.
            return candidate_checkpoint_roots({"ckpt_name": name}, CHECKPOINT_ROOT,
                                              policy_dir=POLICY_ROOT, explicit_keys=(), resolve=False)

        return select_checkpoint(POLICY_ROOT, None if _none_like(value) else value,
                                 bench=cfg.get("bench_name") or cfg.get("dataset_name"),
                                 action_type=cfg.get("action_type") or "joint", seed=cfg.get("seed", 0),
                                 extra_candidates=named_candidates)

    def _build_example(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        # Refresh embodiment-specific URDF context before reading qpos.
        dynamic = obs.get("robot_context")
        if dynamic is not None:
            if not isinstance(dynamic, Mapping):
                raise ValueError("robot_context must contain urdf_paths and joint_names")
            paths = dynamic.get("urdf_paths")
            names = dynamic.get("joint_names")
            if not isinstance(paths, (list, tuple)) or not isinstance(names, (list, tuple)):
                raise ValueError("robot_context needs two urdf_paths and two joint_names lists")
            self.update_robot_context(paths, names)
        elif self._dynamic_urdf_required:
            raise ValueError("Generated EmbodiBench URDF context is required in every observation")
        if self.robot_context is None:
            raise ValueError("No episode URDF context supplied before policy inference")
        state, vision = obs.get("state"), obs.get("vision")
        if not isinstance(state, Mapping) or not isinstance(vision, Mapping):
            raise ValueError("XPolicyLab observation needs state and vision mappings")
        mask = np.asarray(self.robot_context["joint_valid_mask"], dtype=bool)
        angles = np.zeros((2, self.max_joints), dtype=np.float32)
        gripper = np.zeros((2,), dtype=np.float32)
        for arm, side in enumerate(("left", "right")):
            q = np.asarray(state[f"{side}_arm_joint_state"], dtype=np.float32).reshape(-1)
            g = np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float32).reshape(-1)
            count = int(mask[arm].sum())
            if q.shape != (count,) or not np.isfinite(q).all():
                raise ValueError(f"{side} arm state must have {count} finite values, got {q.shape}")
            if g.shape != (1,) or not np.isfinite(g).all() or not 0 <= float(g[0]) <= 1:
                raise ValueError(f"{side} gripper state must be one normalized value in [0,1]")
            angles[arm, :count] = q
            gripper[arm] = g[0] if count else 0.0
        images = []
        for camera_name in self.camera_names:
            xpl_name = _CAMERA_TO_XPL[camera_name]
            camera = vision.get(xpl_name)
            if not isinstance(camera, Mapping) or "color" not in camera:
                raise KeyError(f"Missing vision/{xpl_name}/color")
            images.append(_image(camera["color"], f"vision/{xpl_name}/color"))
        instruction = obs.get("instruction") or obs.get("instructions") or self.default_instruction
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else self.default_instruction
        return {**self.robot_context, "image": images, "lang": str(instruction), "state_angle": angles, "state_gripper": gripper}

    @staticmethod
    def _cpu(value):
        if torch.is_tensor(value): return value.detach().cpu()
        if isinstance(value, list): return [Model._cpu(x) for x in value]
        if isinstance(value, tuple): return tuple(Model._cpu(x) for x in value)
        if isinstance(value, dict): return {k: Model._cpu(v) for k, v in value.items()}
        return value

    @torch.inference_mode()
    def _predict_one(self, obs: Mapping[str, Any]) -> list[dict[str, np.ndarray]]:
        example = self._build_example(obs)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if self.precision == "bf16" and self.device.type == "cuda" else contextlib.nullcontext()
        with context:
            if self.head_device == self.device:
                result = self.model.predict_action(
                    [example],
                    num_inference_steps=self.num_inference_steps,
                    secondary_remask=False if self._df_enabled else self.secondary_remask,
                    secondary_remask_threshold=self.secondary_remask_threshold,
                    continuous_inference_steps=self.continuous_inference_steps)
            else:
                result = self._predict_routed([example])
        result = self._restore_execution_branches(self._cpu(result), example)
        packed_value = result["packed_actions"][0]
        # NumPy has no BF16 dtype; convert only at the simulator boundary.
        packed = (packed_value.detach().float().cpu().numpy() if torch.is_tensor(packed_value)
                  else np.asarray(packed_value, dtype=np.float32))
        if packed.ndim != 2 or not np.isfinite(packed).all():
            raise ValueError(f"Expected finite packed action matrix, got {packed.shape}")
        packed = packed[:self.execution_horizon]
        mask = np.asarray(self.robot_context["joint_valid_mask"], dtype=bool)
        dims = [int(row.sum()) for row in mask]
        expected = sum(dim + 1 for dim in dims if dim)
        if packed.shape[1] != expected:
            raise ValueError(f"Expected packed action width {expected}, got {packed.shape}")
        actions = []
        for row in packed:
            action, offset = {}, 0
            for side, dim in zip(("left", "right"), dims):
                action[f"{side}_arm_joint_state"] = row[offset:offset+dim].copy() if dim else np.empty(0, dtype=np.float32)
                action[f"{side}_ee_joint_state"] = np.asarray([row[offset+dim] if dim else 0.0], dtype=np.float32)
                if dim:
                    offset += dim + 1
            actions.append(action)
        return actions

    def prepare_case(self, case_meta=None):
        if case_meta and case_meta.get("instruction"):
            self.default_instruction = str(case_meta["instruction"])

    def update_obs(self, obs): self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if isinstance(obs_list, Mapping): obs_list = [obs_list]
        if not obs_list: raise ValueError("update_obs_batch received an empty list")
        self._batch = {}
        self._order = []
        for index, obs in enumerate(obs_list):
            env_idx = int(obs.get("env_idx", index))
            self._batch[env_idx] = dict(obs)
            self._order.append(env_idx)

    def get_action(self):
        env_idx = self._order[0] if self._order else 0
        return self.get_action_batch([env_idx])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None: env_idx_list = self._order
        if isinstance(env_idx_list, (int, np.integer)): env_idx_list = [int(env_idx_list)]
        env_idx_list = [int(x) for x in env_idx_list]
        if not env_idx_list: raise ValueError("get_action_batch received no environment indices")
        missing = [x for x in env_idx_list if x not in self._batch]
        if missing: raise ValueError(f"No stored observation for environment indices {missing}")
        return [self._predict_one(self._batch[x]) for x in env_idx_list]

    def reset(self):
        self._batch, self._order = {}, []

    def close(self):
        self._batch, self._order = {}, []
        self.robot_signature, self.robot_context = None, None
        _PolicyRuntime.close(self)


def get_model(usr_args):
    return Model(usr_args)


def reset_model(model):
    model.reset()

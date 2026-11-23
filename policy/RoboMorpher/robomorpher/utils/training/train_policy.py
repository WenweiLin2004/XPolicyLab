#!/usr/bin/env python
"""Native FSDP FULL_SHARD training for selectable discrete-code policy architectures."""
from __future__ import annotations
import argparse, contextlib, hashlib, json, math, os, random, sys
from copy import deepcopy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
# Rich assigns style IDs using Python RNG at first import; initialize logging
# before run_training seeds streams so cold and warm starts share the same seed.
import rich.style  # noqa: F401
from robomorpher.utils.inference_paths import bundled_vlm_suffix
from robomorpher.utils.training.logging import TrainingProgress, log_info, log_train_metrics
from robomorpher.utils.training.performance import TrainingPerformance, log_training_performance
from robomorpher.utils.training.checkpointing import (
    atomic_torch_save, prepare_output_directory, validate_checkpoint_destination,
    register_and_prune_checkpoint, load_weight_checkpoint,
    policy_meta_path,
)


# Frozen tokenizer weights live in robomorpher/pretrained/tokenizer. A policy file
# keeps the trainable policy only; these prefixes are never written again.
_OMITTED_WEIGHT_PREFIXES = ("codec.", "state_tokenizer.", "action_tokenizer.")


class ResumeBatchSampler:
    """Skip consumed index batches without decoding their samples.

    Length stays the logical full epoch length. The training loop enumerates
    from start_batch, preserving accumulation and final-partial-batch indices.
    """
    def __init__(self, base):
        self.base = base
        self.start_batch = 0

    def __iter__(self):
        from itertools import islice
        return islice(iter(self.base), self.start_batch, None)

    def __len__(self):
        return len(self.base)


def policy_weight_path(output, step):
    """Spark-style weight file: <run>/checkpoints/steps_<N>_pytorch_model.pt."""
    return Path(output) / "checkpoints" / f"steps_{int(step)}_pytorch_model.pt"


def wandb_enabled(config):
    tracking = config.get("wandb", {})
    if "enabled" in tracking and not isinstance(tracking["enabled"], bool):
        raise ValueError("wandb.enabled must be a boolean")
    return bool(tracking.get("enabled", False))


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _consume_trainer_aliases(cfg):
    """Accept Spark's trainer.learning_rate map and the other upstream aliases."""
    tc = cfg["trainer"]
    if "learning_rate" in tc:
        rates = tc["learning_rate"]
        needed = {"base", "qwen_vl_interface", "action_model"}
        if not isinstance(rates, dict) or set(rates) != needed:
            raise ValueError("trainer.learning_rate requires base, qwen_vl_interface, and action_model")
        base, interface, action = (float(rates["base"]), float(rates["qwen_vl_interface"]),
                                   float(rates["action_model"]))
        if interface != base:
            raise ValueError("qwen_vl_interface shares the backbone group and must equal base")
        if "lr" in tc and float(tc["lr"]) != base:
            raise ValueError("trainer.lr conflicts with learning_rate.base")
        if "head_lr" in tc and float(tc["head_lr"]) != action:
            raise ValueError("trainer.head_lr conflicts with learning_rate.action_model")
        tc["lr"], tc["head_lr"] = base, action
        del tc["learning_rate"]
    if "optimizer" in tc:
        optimizer = tc["optimizer"]
        if not isinstance(optimizer, dict) or optimizer.get("name", "AdamW") != "AdamW":
            raise ValueError("trainer.optimizer.name must be AdamW")
        for key in ("betas", "eps", "weight_decay"):
            if key in optimizer:
                tc[key] = optimizer[key]
        del tc["optimizer"]
    if "save_interval" in tc:
        if "save_every_steps" in tc and tc["save_every_steps"] != tc["save_interval"]:
            raise ValueError("trainer.save_every_steps conflicts with save_interval")
        tc["save_every_steps"] = tc.pop("save_interval")
    if "logging_frequency" in tc:
        if "log_every_steps" in tc and tc["log_every_steps"] != tc["logging_frequency"]:
            raise ValueError("trainer.log_every_steps conflicts with logging_frequency")
        tc["log_every_steps"] = tc.pop("logging_frequency")
    if "gradient_clipping" in tc:
        if "max_grad_norm" in tc and float(tc["max_grad_norm"]) != float(tc["gradient_clipping"]):
            raise ValueError("trainer.max_grad_norm conflicts with gradient_clipping")
        tc["max_grad_norm"] = tc.pop("gradient_clipping")
    if "max_train_steps" in tc:
        if "total_steps" in tc or "max_steps" in tc:
            raise ValueError("trainer.max_train_steps conflicts with total_steps or max_steps")
        tc["total_steps"] = tc.pop("max_train_steps")
    if "repeated_diffusion_steps" in tc:
        steps = tc.pop("repeated_diffusion_steps")
        action = cfg.setdefault("framework", {}).setdefault("action_model", {})
        if "repeated_diffusion_steps" in action and action["repeated_diffusion_steps"] != steps:
            raise ValueError("repeated_diffusion_steps conflicts with framework.action_model")
        action["repeated_diffusion_steps"] = steps
    if "enable_gradient_checkpointing" in tc:
        enabled = bool(tc.pop("enable_gradient_checkpointing"))
        existing = cfg.get("qwenvl", {}).get("gradient_checkpointing")
        if existing is not None and bool(existing) != enabled:
            raise ValueError("enable_gradient_checkpointing conflicts with qwenvl.gradient_checkpointing")
        framework = cfg.setdefault("framework", {})
        framework.setdefault("qwenvl", {})["gradient_checkpointing"] = enabled
        framework.setdefault("action_model", {})["gradient_checkpointing"] = enabled
    if "enable_mixed_precision_training" in tc:
        enabled = bool(tc.pop("enable_mixed_precision_training"))
        precision = "bf16" if enabled else "fp32"
        if "precision" in tc and tc["precision"] != precision:
            raise ValueError("enable_mixed_precision_training conflicts with trainer.precision")
        tc["precision"] = precision
    leftover = {"mixed_precision", "mixedprecision"} & set(tc)
    if leftover:
        raise ValueError(f"Removed trainer settings are no longer supported: {sorted(leftover)}")


def normalize_training_config(config):
    """Validate the trainer schema and accept Spark's learning_rate groups."""
    cfg = _canonical(OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config)
    tc = cfg.get("trainer")
    if not isinstance(tc, dict):
        raise ValueError("trainer mapping is required")
    _consume_trainer_aliases(cfg)
    budget = tc.get("total_steps")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
        raise ValueError("trainer.total_steps must be a positive integer or null")
    return cfg


def _optimizer_scheduler_settings(config, total_steps):
    """Validate effective AdamW and per-group cosine floors before model updates."""
    betas = config.get("betas", [.9, .999])
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("trainer.betas must contain two numbers in [0,1)")
    betas = [float(v) for v in betas]
    if any(not math.isfinite(v) or not 0 <= v < 1 for v in betas):
        raise ValueError("trainer.betas must contain two numbers in [0,1)")
    eps, weight_decay = float(config.get("eps", 1e-8)), float(config.get("weight_decay", .01))
    if not math.isfinite(eps) or eps <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("AdamW eps must be positive and weight_decay nonnegative, both finite")
    lrs = {"backbone_and_conditioning": float(config["lr"]),
           "action_head": float(config.get("head_lr", config["lr"]))}
    if any(not math.isfinite(v) or v <= 0 for v in lrs.values()):
        raise ValueError("Learning rates must be finite and positive")
    implementation = config.get("scheduler_implementation", "starvla")
    if not isinstance(implementation, str) or implementation not in {"per_group", "starvla"}:
        raise ValueError("scheduler_implementation must be per_group or starvla")
    if implementation == "starvla" and "head_min_lr" in config:
        raise ValueError("starvla uses one shared LR ratio; head_min_lr requires the scheduler")
    schedule = str(config.get("lr_scheduler_type", "cosine"))
    if schedule not in {"cosine", "cosine_with_min_lr"}:
        raise ValueError("lr_scheduler_type must be cosine or cosine_with_min_lr")
    kwargs = dict(config.get("scheduler_specific_kwargs") or {})
    if set(kwargs) - {"min_lr", "min_lr_rate"}:
        raise ValueError("Unsupported scheduler_specific_kwargs; use min_lr or min_lr_rate")
    if "min_lr" in config:
        if "min_lr" in kwargs and float(kwargs["min_lr"]) != float(config["min_lr"]):
            raise ValueError("Conflicting trainer.min_lr and scheduler_specific_kwargs.min_lr")
        kwargs["min_lr"] = config["min_lr"]
    if "min_lr" in kwargs and "min_lr_rate" in kwargs:
        raise ValueError("Specify only one of min_lr and min_lr_rate")
    if schedule == "cosine" and (kwargs or "head_min_lr" in config):
        raise ValueError("A minimum LR requires lr_scheduler_type=cosine_with_min_lr")
    if schedule == "cosine_with_min_lr" and not kwargs:
        raise ValueError("cosine_with_min_lr requires min_lr or min_lr_rate")
    floors = {name: 0. for name in lrs}
    if "min_lr_rate" in kwargs:
        rate = float(kwargs["min_lr_rate"])
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("min_lr_rate must be finite and within [0,1]")
        floors = {name: lr * rate for name, lr in lrs.items()}
    elif "min_lr" in kwargs:
        if implementation == "starvla":
            # HF converts min_lr / optimizer.defaults['lr'] to ONE multiplier.
            # The official base=1e-5, head=1e-4, min_lr=5e-7 recipe therefore
            # ends at 5e-7 / 5e-6; it does not impose an equal absolute floor.
            rate = float(kwargs["min_lr"]) / lrs["backbone_and_conditioning"]
            floors = {name: lr * rate for name, lr in lrs.items()}
        else:
            floors = {name: float(kwargs["min_lr"]) for name in lrs}
    if "head_min_lr" in config: floors["action_head"] = float(config["head_min_lr"])
    if any(not math.isfinite(v) or not 0 <= v <= lrs[name] for name,v in floors.items()):
        raise ValueError("Each minimum LR must be finite and within [0, group learning rate]")
    if "num_warmup_steps" in config:
        raw_steps = config["num_warmup_steps"]
        warmup_steps = int(raw_steps)
        if isinstance(raw_steps, bool) or warmup_steps != raw_steps:
            raise ValueError("num_warmup_steps must be an integer")
    else:
        ratio = float(config.get("warmup_ratio", .05))
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("warmup_ratio must be within [0,1]")
        warmup_steps = round(total_steps * ratio)
    if not 0 <= warmup_steps <= total_steps:
        raise ValueError("num_warmup_steps must be within [0,total_steps]")
    optimizer = {"name": "AdamW", "betas": betas, "eps": eps, "weight_decay": weight_decay}
    scheduler = {"name": schedule, "min_lrs": floors}
    if implementation == "starvla":
        optimizer["fused"] = True
        scheduler.update(implementation="starvla", kwargs={key: float(value) for key,value in kwargs.items()})
    # Do not add implementation keys to contracts: old checkpoints keep
    # their exact scheduler and strict-resume contract without implicit migration.
    return {"optimizer": optimizer, "scheduler": scheduler, "warmup_steps": warmup_steps}


def split_episode_indices(dataset, fraction, seed):
    """Split complete episodes; overlapping chunks never cross the split."""
    fraction = float(fraction)
    if not 0 <= fraction < 1:
        raise ValueError("data.validation_split must be in [0,1)")
    count = len(dataset.episodes)
    if not count or (fraction and count < 2):
        raise ValueError("Validation requires at least two episodes")
    order = np.random.default_rng(int(seed)).permutation(count)
    nval = min(count - 1, max(1, round(count * fraction))) if fraction else 0
    validation = set(order[:nval].tolist())
    train, val, previous = [], [], 0
    for idx, end in enumerate(dataset._ends):
        (val if idx in validation else train).extend(range(previous, end))
        previous = end
    manifest = [{"path": str(ep["path"]), "length": int(ep["length"]), "validation": i in validation}
                for i, ep in enumerate(dataset.episodes)]
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return train, val, {"sha256": digest, "train_episodes": count-nval,
                        "validation_episodes": nval, "episodes": manifest}


def _distributed(config):
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    mode = str(config.get("distributed", "fsdp" if world > 1 else "single"))
    if world > 1 and mode != "fsdp":
        raise ValueError("Multi-GPU requires native FSDP; DDP is not a memory fallback")
    if config.get("device", "cuda") == "cpu":
        if world > 1: raise ValueError("Production FSDP requires CUDA")
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if world > 1 and not dist.is_initialized(): dist.init_process_group("nccl")
    return rank, world, device


def _codec_contract(model):
    if any(p.requires_grad for p in model.codec.parameters()):
        raise ValueError("Action codec must be frozen")
    for tensor in list(model.codec.parameters()) + list(model.codec.buffers()):
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError("Action codec must remain FP32")
    model.codec.eval()
    spec = _canonical(model.codec.spec.to_dict())
    if not spec.get("fingerprint"): raise ValueError("Codec source fingerprint is required")
    return spec


def _wrap_fsdp(model, device, precision):
    from functools import partial
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    block_types = {type(m) for m in model.modules() if type(m).__name__ in {
        "Qwen2_5_VLDecoderLayer", "Qwen3VLTextDecoderLayer", "Qwen3VLMoeTextDecoderLayer",
        "BasicTransformerBlock"}}
    if not block_types: raise ValueError("No supported Qwen or action blocks for FSDP wrapping")
    # Discrete Forcing aliases the first N continuous-refiner blocks to the
    # categorical action DiT. If auto-wrap sees the same module through both
    # parents, one reference can remain a raw sharded module and fail in
    # F.linear. Keep aliased blocks out of nested wrappers; root FSDP still
    # shards their parameters once and both references use the same module.
    shared_ids = set()
    action_model = getattr(model, "action_model", None)
    continuous_refiner = getattr(model, "continuous_refiner", None)
    action_blocks = getattr(getattr(action_model, "model", None), "transformer_blocks", None)
    continuous_blocks = getattr(getattr(continuous_refiner, "model", None), "transformer_blocks", None)
    if action_blocks is not None and continuous_blocks is not None:
        action_blocks = list(action_blocks)
        shared_ids = {
            id(block) for block in continuous_blocks
            if any(block is action_block for action_block in action_blocks)
        }
    transformer_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls=block_types
    )
    def wrap_policy(module, recurse, nonwrapped_numel):
        if id(module) in shared_ids:
            return False
        return transformer_policy(module, recurse, nonwrapped_numel)
    mp = (MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                         buffer_dtype=torch.float32, cast_forward_inputs=False)
          if precision == "bf16" else None)
    return FSDP(model, auto_wrap_policy=wrap_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD, ignored_modules=[model.codec],
                use_orig_params=True, mixed_precision=mp, device_id=device, sync_module_states=True,
                limit_all_gathers=True)


def _parameter_groups(model, config):
    """Separate action backbones from lower-LR conditioning adapters.

    URDF modulation/projectors can be nested inside either action branch, so
    these exceptions take precedence over the containing action-model name.
    Native DiT time conditioning remains part of the action model. Module-path
    components also work with FSDP wrapper prefixes (use_orig_params=True).
    """
    base, head = [], []
    base_modules = {
        "vl_projectors", "urdf_projector", "urdf_modulation", "urdf_modulations",
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parts = name.split(".")
        conditioning = bool(base_modules.intersection(parts))
        action_branch = ("action_model" in parts or "continuous_refiner" in parts
                         or ("architecture" in parts and "action_backbone" in parts))
        (head if action_branch and not conditioning else base).append(parameter)
    # named_parameters deduplicates DF's shared blocks/future tokens by identity.
    if not base and not head:
        raise ValueError("No trainable policy parameters")
    groups = []
    if base:
        groups.append({"params": base, "lr": float(config["lr"]),
                       "name": "backbone_and_conditioning"})
    if head:
        groups.append({"params": head, "lr": float(config.get("head_lr", config["lr"])),
                       "name": "action_head"})
    return groups


def _lr_factor(step, *, total_steps, warmup_steps, min_lr_ratio=0.):
    if warmup_steps and step < warmup_steps: return (step + 1) / warmup_steps
    progress = min(1., max(0., (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return min_lr_ratio + (1 - min_lr_ratio) * .5 * (1 + math.cos(math.pi * progress))


def _build_lr_scheduler(optimizer, config, total_steps, settings):
    """Keep resumes exact; the opt-in profile calls starVLA's HF factory."""
    schedule = settings["scheduler"]
    if schedule.get("implementation") == "starvla":
        from transformers import get_scheduler
        if optimizer.defaults["lr"] != float(config["lr"]):
            raise ValueError("starvla scheduler requires optimizer.defaults.lr == trainer.lr")
        return get_scheduler(
            name=schedule["name"], optimizer=optimizer,
            num_warmup_steps=settings["warmup_steps"], num_training_steps=total_steps,
            scheduler_specific_kwargs=schedule["kwargs"],
        )
    from functools import partial
    return LambdaLR(optimizer, [partial(
        _lr_factor, total_steps=total_steps, warmup_steps=settings["warmup_steps"],
        min_lr_ratio=schedule["min_lrs"][group["name"]] / group["lr"],
    ) for group in optimizer.param_groups])


def _state_context(model, fsdp):
    if not fsdp: return contextlib.nullcontext()
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, StateDictType,
                                       FullStateDictConfig, FullOptimStateDictConfig)
    return FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                               FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                               FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True))


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state(state["cuda"])



def _rank0_error(error, world):
    if world > 1:
        data = [error]
        dist.broadcast_object_list(data, src=0)
        error = data[0]
    if error is not None:
        raise ValueError(error)


def _checkpoint_destination(path, step):
    """Return (run directory, weight file)."""
    path = Path(path)
    if path.suffix != ".pt":
        path = policy_weight_path(path, step)
    if path.parent.name != "checkpoints":
        raise ValueError("Policy weight files belong in <output>/checkpoints/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent.parent, path


def _export_weight(key, value, precision):
    tensor = value.detach().cpu()
    if precision == "bf16" and tensor.is_floating_point() and not key.startswith(_OMITTED_WEIGHT_PREFIXES):
        return tensor.to(dtype=torch.bfloat16)
    return tensor


def _save_checkpoint(model, optimizer, scheduler, *, path, rank, world, fsdp,
                     codec_spec, contract, epoch, next_batch, step, config, save_training_state=False):
    entry_rng = _rng_state()
    try:
        # Normal exports remain weight-only. An explicit checkpoint+stop also
        # saves each rank's FP32 master shards and Adam state for exact continuation.
        training_state_relative = None
        if save_training_state:
            state_path = Path(path).parent / "training_state" / f"steps_{int(step)}" / f"rank_{rank}.pth"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save({
                "parameters": {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad},
                "optimizer": optimizer.state_dict(),
            }, state_path)
            training_state_relative = str(state_path.relative_to(Path(path).parent))
        # Every FSDP rank enters the state-dict collective; only rank0 persists weights.
        with _state_context(model, fsdp):
            full = model.state_dict()
        rng = entry_rng
        if world > 1:
            gathered = [None] * world if rank == 0 else None
            dist.gather_object(rng, gathered, dst=0)
        else: gathered = [rng]
        error = None
        if rank == 0:
            try:
                output, weight_path = _checkpoint_destination(path, step)
                validate_checkpoint_destination(output, weight_path, contract)
                precision = config.get("trainer", {}).get("precision", "fp32")
                state = {
                    key: _export_weight(key, value, precision)
                    for key, value in full.items()
                    if not key.startswith(_OMITTED_WEIGHT_PREFIXES)
                }
                meta = {"codec_spec": codec_spec, "contract": contract, "scheduler": scheduler.state_dict(),
                        "epoch": epoch, "next_batch": next_batch, "global_step": step,
                        "world_size": world, "rng_by_rank": gathered, "config": config}
                if save_training_state:
                    meta["training_state_pattern"] = "training_state/steps_" + str(int(step)) + "/rank_{rank}.pth"
                atomic_torch_save(state, weight_path)
                atomic_torch_save(meta, policy_meta_path(weight_path))
                register_and_prune_checkpoint(output, weight_path, contract, 0, allow_unlimited=True, checkpoint_counter=step)
                log_info(task="policy", phase="checkpoint", step=step, epoch=epoch, path=weight_path)
            except Exception as exc:
                error = "Checkpoint save failed: " + str(exc)
        _rank0_error(error, world)
        if world > 1: dist.barrier()
    finally:
        # Checkpoint serialization/collectives must not advance any training RNG.
        _restore_rng(entry_rng)


def load_training_checkpoint(path):
    """Return metadata plus the weight state_dict under the key ``model``."""
    weights, meta = load_weight_checkpoint(path)
    if any(name.startswith(_OMITTED_WEIGHT_PREFIXES) for name in weights):
        raise ValueError("Policy checkpoints must not include tokenizer weights")
    return {"model": weights, **meta}


def _resume_contract_view(contract):
    """Canonicalize only the bundled VLM identity fields for resume comparison."""
    if not isinstance(contract, dict):
        return contract
    view = deepcopy(contract)
    framework = view.get("framework")
    fields = [(view.get("qwenvl"), "base_vlm"), (view.get("model_contract"), "backbone_id")]
    if isinstance(framework, dict):
        fields.append((framework.get("qwenvl"), "base_vlm"))
    for section, key in fields:
        if not isinstance(section, dict) or not isinstance(section.get(key), str):
            continue
        path = Path(section[key])
        if path.is_absolute():
            # Compare mount and symlink aliases of this checkout without adopting
            # a model under an unrelated directory with the same basename.
            for parent in path.parents:
                relative = path.relative_to(parent)
                if bundled_vlm_suffix(relative) and parent.resolve() == ROOT:
                    path = relative
                    break
        if not path.parts or path.parts[0] not in {"robomorpher", "checkpoints"}:
            continue
        suffix = bundled_vlm_suffix(path)
        if suffix:
            section[key] = f"robomorpher/pretrained/vlm/{suffix}"
    return view


def _load_checkpoint(model, optimizer, scheduler, *, path, rank, world, fsdp, codec_spec, contract):
    weights, checkpoint = load_weight_checkpoint(path)
    if checkpoint.get("codec_spec") != codec_spec:
        raise ValueError("Checkpoint codec spec/fingerprint differs; refusing silent tokenizer substitution")
    saved_contract = checkpoint.get("contract")
    if saved_contract != contract and _resume_contract_view(saved_contract) != _resume_contract_view(contract):
        raise ValueError("Resume model/data/LR plan differs; create a new experiment")
    if checkpoint.get("world_size") != world:
        raise ValueError("RNG resume requires the same world size")
    if fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        context = FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=True, rank0_only=False))
    else: context = contextlib.nullcontext()
    state = dict(weights)
    if any(name.startswith(_OMITTED_WEIGHT_PREFIXES) for name in state):
        raise ValueError("Policy checkpoints must not override the separately fingerprinted frozen codec")
    # Saved policy weights omit the frozen codec. Its checkpoint/fingerprint has
    # already been verified above; supply its live state for recursive tokenizer
    # schema hooks, which require a complete state even with strict=False.
    state.update({f"codec.{name}": value for name, value in model.codec.state_dict().items()})
    with context:
        live = model.state_dict()
        for name, value in list(state.items()):
            current = live.get(name)
            if torch.is_tensor(value) and torch.is_tensor(current) and value.dtype != current.dtype:
                state[name] = value.to(dtype=current.dtype)
        result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or result.missing_keys:
        raise RuntimeError(f"Policy checkpoint key mismatch: {result}")
    pattern = checkpoint.get("training_state_pattern")
    if pattern:
        parent = Path(path).parent.resolve()
        training_path = (parent / pattern.format(rank=rank)).resolve()
        if not training_path.is_relative_to(parent / "training_state"):
            raise ValueError("Training-state shard path escapes checkpoint directory")
        local_state = torch.load(training_path, map_location="cpu", weights_only=False)
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in local_state["parameters"].items():
                if name not in parameters or parameters[name].shape != value.shape:
                    raise ValueError("Training-state parameter shard differs: " + name)
                parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))
        optimizer.load_state_dict(local_state["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    # Weight-only exports have no optimizer param groups to restore LR.
    for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
        group["lr"] = lr
    _restore_rng(checkpoint["rng_by_rank"][rank])
    # The manifest and all subsequent checkpoints belong to the original run.
    return checkpoint["epoch"], checkpoint["next_batch"], checkpoint["global_step"], saved_contract


def _autocast(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16")


_AUX_METRIC_KEYS = ("ce_loss", "ce_loss_weighted", "continuous_flow_loss", "continuous_one_step_loss", "continuous_loss", "continuous_loss_weighted")


@torch.no_grad()
def evaluate(model, dataset, indices, *, batch_size, rank, world, device, precision, seed,
             return_metrics=False):
    """Exact validation coverage; exhausted FSDP ranks still enter dummy forwards."""
    if not indices: return None
    from robomorpher.utils.data.load_data.policy_dataset import collate_policy_samples
    loader = DataLoader(Subset(dataset, indices[rank::world]), batch_size=batch_size, num_workers=0,
                        collate_fn=collate_policy_samples, generator=torch.Generator().manual_seed(seed+rank))
    rounds = torch.tensor(len(loader), device=device, dtype=torch.long)
    if world > 1: dist.all_reduce(rounds, op=dist.ReduceOp.MAX)
    rng = _rng_state(); torch.manual_seed(seed+rank)
    if device.type == "cuda": torch.cuda.manual_seed(seed+rank)
    model.eval(); iterator = iter(loader); total = torch.zeros(2, device=device, dtype=torch.float64)
    component_totals = {}
    for _ in range(int(rounds.item())):
        examples = next(iterator, None); count = 0 if examples is None else len(examples)
        if examples is None: examples = [dataset[indices[0]]]
        with _autocast(device, precision):
            loss_output = model(examples)
            loss = loss_output["action_loss"]
        if count: total[0] += loss.detach().double()*count; total[1] += count
        for key in _AUX_METRIC_KEYS:
            if key in loss_output:
                component_totals.setdefault(key, torch.zeros((), device=device, dtype=torch.float64))
                if count: component_totals[key] += loss_output[key].detach().double()*count
    if world > 1:
        dist.all_reduce(total)
        for key in sorted(component_totals): dist.all_reduce(component_totals[key])
    _restore_rng(rng); model.train()
    result = {"action_loss": (total[0]/total[1].clamp_min(1)).item()}
    result.update({key: (value/total[1].clamp_min(1)).item() for key,value in component_totals.items()})
    return result if return_metrics else result["action_loss"]


def run_training(config, *, model_factory=None, dataset_factory=None):
    """Injectable factories permit deterministic, tiny-model integration tests."""
    cfg = normalize_training_config(config)
    tracking_enabled = wandb_enabled(cfg)
    tc = cfg["trainer"]; rank, world, device = _distributed(tc); fsdp = world > 1
    precision = str(tc.get("precision", "bf16"))
    if precision not in {"bf16", "fp32"}: raise ValueError("precision must be bf16 or fp32")
    seed = int(tc.get("seed", 42))
    random.seed(seed+rank); np.random.seed(seed+rank); torch.manual_seed(seed+rank)
    if device.type == "cuda": torch.cuda.manual_seed(seed+rank)
    if model_factory is None:
        from robomorpher.model.policy_factory import build_policy
        model_factory = build_policy
    model = model_factory(cfg)
    # Homogeneous FP32 master parameters prevent mixed-dtype FSDP flat groups.
    # Convert on CPU, then let each auto-wrapped block move/shard incrementally.
    for name, child in model.named_children():
        if name != "codec": child.float()
    codec_spec = _codec_contract(model)
    if fsdp: model.codec.to(device)
    else: model.to(device)
    model_contract = _canonical(model.checkpoint_contract()) if hasattr(model, "checkpoint_contract") else None
    qwenvl_config = cfg["framework"].get("qwenvl", cfg.get("qwenvl", {}))
    if qwenvl_config.get("gradient_checkpointing", True):
        backbone = getattr(model, "backbone", None); hf = getattr(backbone, "model", backbone)
        if hf is not None and hasattr(hf, "gradient_checkpointing_enable"):
            hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(hf, "config"): hf.config.use_cache = False
    if dataset_factory is None:
        dc = cfg["data"]
        data_format = str(dc.get("format", dc.get("type", "lerobot"))).lower()
        action_horizon = int(getattr(model, "action_horizon",
            cfg["framework"]["action_model"].get("action_horizon", 50)))
        if data_format in {"lerobot", "lerobot_v3", "policy"}:
            from robomorpher.utils.data.load_data.policy_dataset import LeRobotPolicyDataset
            dataset_options = {}
            if "joint_auxiliary" in dc:
                if not isinstance(dc["joint_auxiliary"], bool):
                    raise ValueError("data.joint_auxiliary must be boolean")
                if dc["joint_auxiliary"]:
                    dataset_options["joint_auxiliary"] = True
            dataset = LeRobotPolicyDataset(
                dc["root"], num_arms=codec_spec["num_arms"],
                max_num_joints=codec_spec["max_num_joints"],
                action_chunk_size=action_horizon,
                camera_names=dc.get("camera_names", ["cam_high"]),
                episode_files=dc.get("episode_files"),
                action_type=dc.get("action_type", "joint"),
                **dataset_options,
            )
        else:
            raise ValueError(
                f"Unsupported data.format={data_format!r}; convert source data to lerobot first"
            )
    else:
        dataset = dataset_factory(cfg)
    dataset_contract = _canonical(dataset.checkpoint_contract()) if hasattr(dataset, "checkpoint_contract") else None
    if rank == 0:
        log_info(task="policy", phase="data", windows=len(dataset), episodes=len(dataset.episodes),
                 skipped_episodes=len(getattr(dataset, "skipped_episodes", [])),
                 action_horizon=int(getattr(model, "action_horizon", 50)),
                 validation_split=cfg["data"].get("validation_split", 0.0))
    data_seed = int(cfg["data"].get("seed", seed))
    train_ids, val_ids, split = split_episode_indices(dataset, cfg["data"].get("validation_split", 0.0), data_seed)
    batch_size, accumulation = int(tc.get("batch_size", 1)), int(tc.get("gradient_accumulation_steps", 1))
    epochs, max_grad_norm = int(tc.get("epochs", 10)), float(tc.get("max_grad_norm", 1.))
    if min(batch_size, accumulation, epochs) < 1: raise ValueError("Batch/accumulation/epochs must be positive")
    train_subset = Subset(dataset, train_ids)
    sampler_config = cfg["data"].get("sampler", {})
    if sampler_config is None:
        sampler_config = {}
    if not hasattr(sampler_config, "get"):
        raise ValueError("data.sampler must be a mapping")
    replacement = sampler_config.get("replacement", cfg["data"].get("replacement", False))
    if not isinstance(replacement, bool):
        raise ValueError("data.sampler.replacement must be boolean")
    requested_samples = sampler_config.get("num_samples", None)
    if requested_samples is not None:
        if (isinstance(requested_samples, bool)
                or int(requested_samples) != requested_samples
                or int(requested_samples) < 1):
            raise ValueError("data.sampler.num_samples must be a positive integer or null")
        requested_samples = int(requested_samples)
    if replacement:
        from robomorpher.utils.data.load_data.policy_dataset import DistributedReplacementSampler
        sampler = DistributedReplacementSampler(
            train_subset, num_replicas=world, rank=rank,
            num_samples=requested_samples, seed=seed,
        )
    else:
        sampler = DistributedSampler(train_subset, num_replicas=world, rank=rank,
                                     shuffle=True, seed=seed, drop_last=True)
    from robomorpher.utils.data.load_data.policy_dataset import collate_policy_samples
    batch_sampler = ResumeBatchSampler(torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False))
    loader = DataLoader(train_subset, batch_sampler=batch_sampler,
        num_workers=int(cfg["data"].get("num_workers", 8)), collate_fn=collate_policy_samples,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed+rank))
    steps_per_epoch = math.ceil(len(loader)/accumulation)
    if steps_per_epoch < 1:
        raise ValueError("The training loader must contain at least one batch")
    if tc.get("total_steps") is None:
        total_steps = epochs*steps_per_epoch
    else:
        # A step budget owns both the scheduler horizon and optimizer-update
        # count. Epochs become a sufficient number of data passes, not a cap.
        total_steps = tc["total_steps"]
        epochs = math.ceil(total_steps/steps_per_epoch)
    training_settings = _optimizer_scheduler_settings(tc, total_steps)
    warmup_steps = training_settings["warmup_steps"]
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if fsdp: model = _wrap_fsdp(model, device, precision)
    adam = training_settings["optimizer"]
    optimizer = AdamW(_parameter_groups(model, tc), lr=float(tc["lr"]),
                      weight_decay=adam["weight_decay"], betas=tuple(adam["betas"]), eps=adam["eps"],
                      **({"fused": adam["fused"]} if "fused" in adam else {}))
    scheduler = _build_lr_scheduler(optimizer, tc, total_steps, training_settings)
    contract = _canonical({"framework": cfg["framework"], "qwenvl": qwenvl_config,
        "model_contract": model_contract, "codec_spec": codec_spec, "split_sha256": split["sha256"], "seed": seed,
        "data_contract": {k:v for k,v in cfg["data"].items() if k != "num_workers"}, "data_seed": data_seed,
        "sampling": {"replacement": replacement,
                     "global_num_samples": (sampler.global_num_samples if replacement else len(train_subset)),
                     "per_rank_num_samples": len(sampler)},
        "dataset_contract": dataset_contract,
        "batch_size": batch_size, "accumulation": accumulation, "total_steps": total_steps,
        "warmup_steps": warmup_steps, "lr": tc["lr"], "head_lr": tc.get("head_lr", tc["lr"]),
        "weight_decay": adam["weight_decay"], "optimizer": adam, "scheduler": training_settings["scheduler"],
        "max_grad_norm": max_grad_norm, "precision": precision})
    if not tc.get("output_dir"):
        raise ValueError("Missing trainer.output_dir for policy checkpoints")
    output = Path(tc["output_dir"]).expanduser().resolve()
    log_every = int(tc.get("log_every_steps", 1))
    if log_every < 1:
        raise ValueError("trainer.log_every_steps must be >= 1")
    epoch_start = next_batch = step = 0
    if tc.get("resume"):
        epoch_start, next_batch, step, contract = _load_checkpoint(model, optimizer, scheduler, path=tc["resume"],
            rank=rank, world=world, fsdp=fsdp, codec_spec=codec_spec, contract=contract)
    error = None
    if rank == 0:
        try:
            prepare_output_directory(output, tc.get("resume"), contract, resume_counter=step)
            (output/"config.json").write_text(json.dumps(cfg, indent=2))
            (output/"data_split.json").write_text(json.dumps(split, indent=2))
            (output/"codec_spec.json").write_text(json.dumps(codec_spec, indent=2))
        except Exception as exc:
            error = str(exc)
    _rank0_error(error, world)
    log_info(task="policy", phase="setup", rank=rank, step=step, epochs=epochs, total_steps=total_steps,
             effective_batch=batch_size*world*accumulation, precision=precision,
             repeated_diffusion_steps=getattr(model, "repeated_diffusion_steps", 1),
             head_batch_per_gpu=(
                batch_size * (int(model.repeated_diffusion_steps) // 2)
                if getattr(model, "paired_phase_sampling", False)
                else batch_size * getattr(model, "repeated_diffusion_steps", 1)),
             sampler_replacement=replacement,
             sampler_global_num_samples=(sampler.global_num_samples if replacement else len(train_subset)),
             save_every_steps=int(tc.get("save_every_steps", 500)), output_dir=output)
    wandb_run = None
    if rank == 0 and tracking_enabled:
        # A resumed process initializes tracking again after loading RNG state.
        # External SDK imports/init must not advance the training random stream.
        tracking_rng = _rng_state()
        try:
            import wandb
            wandb_dir = Path(cfg["wandb"].get("dir", "outputs/policy/wandb")).expanduser()
            wandb_dir.mkdir(parents=True, exist_ok=True)
            wandb_run = wandb.init(dir=str(wandb_dir), project=cfg["wandb"].get("project", "RoboMorpher"),
                                  name=cfg["wandb"].get("name"), config=cfg)
        finally:
            _restore_rng(tracking_rng)
    # max_steps is an interruption/smoke cap, not a silent change to the LR schedule.
    max_steps = min(total_steps, int(tc.get("max_steps") or total_steps))
    requested_stop = False
    save_every = int(tc.get("save_every_steps", 500))
    model.train(); optimizer.zero_grad(set_to_none=True); last_val = None
    saved_epoch, saved_next = epoch_start, next_batch
    progress = TrainingProgress(total=total_steps, initial=step, rank=rank)
    performance = TrainingPerformance(warmup_updates=10)
    try:
        for epoch in range(epoch_start, epochs):
            if step >= max_steps: break
            sampler.set_epoch(epoch); window_loss = torch.zeros((), device=device)
            window_metrics = {}
            performance.start_fetch()
            batch_sampler.start_batch = next_batch if epoch == epoch_start else 0
            for batch_index, examples in enumerate(loader, start=batch_sampler.start_batch):
                if step >= max_steps: break
                performance.batch_ready(len(examples))
                start = (batch_index//accumulation)*accumulation
                actual_accumulation = min(accumulation, len(loader)-start)
                sync = batch_index+1 == len(loader) or (batch_index+1)%accumulation == 0
                with (model.no_sync() if fsdp and not sync else contextlib.nullcontext()):
                    with _autocast(device, precision):
                        loss_output = model(examples)
                        loss = loss_output["action_loss"]
                    finite = torch.tensor(int(torch.isfinite(loss).all()), device=device)
                    if world > 1: dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item(): raise FloatingPointError("Nonfinite policy loss")
                    (loss/actual_accumulation).backward()
                window_loss += loss.detach().float()/actual_accumulation
                for key in _AUX_METRIC_KEYS:
                    if key in loss_output:
                        window_metrics.setdefault(key, torch.zeros((), device=device))
                        window_metrics[key] += loss_output[key].detach().float()/actual_accumulation
                if not sync:
                    performance.finish_microbatch()
                    performance.start_fetch()
                    continue
                grad = (model.clip_grad_norm_(max_grad_norm) if fsdp else
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
                finite = torch.tensor(int(torch.isfinite(grad)), device=device)
                if world > 1: dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item(): raise FloatingPointError("Nonfinite policy gradient")
                used_lrs = [group["lr"] for group in optimizer.param_groups]
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
                saved_epoch, saved_next = ((epoch+1, 0) if batch_index+1 == len(loader) else (epoch, batch_index+1))
                if world > 1:
                    dist.all_reduce(window_loss)
                    for key in sorted(window_metrics): dist.all_reduce(window_metrics[key])
                logs = {"train/action_loss": window_loss.item()/world, "train/grad_norm": float(grad),
                        "train/global_step": step, **{f"train/lr_group_{i}": lr for i,lr in enumerate(used_lrs)}}
                logs.update({f"train/{key}": value.item()/world for key,value in window_metrics.items()})
                performance.finish_microbatch()
                performance.complete_update(step)
                log_due = step % log_every == 0 or step == 1
                if log_due:
                    log_training_performance(performance, output=output, device=device, rank=rank,
                                             world=world, step=step, epoch=epoch+1)
                progress.update(data_times=performance.last_update["data_times"],
                                model_times=performance.last_update["model_times"])
                if rank == 0:
                    with (output/"metrics.jsonl").open("a") as stream: stream.write(json.dumps(logs)+"\n")
                    if log_due:
                        console_metrics = {"action_loss": logs["train/action_loss"],
                            **{key: logs[f"train/{key}"] for key in _AUX_METRIC_KEYS if f"train/{key}" in logs},
                            "grad_norm": float(grad),
                            **{f"learning_rate/{group['name']}": lr
                               for group, lr in zip(optimizer.param_groups, used_lrs)},
                            "epoch": round(epoch + (batch_index+1)/len(loader), 2)}
                        log_train_metrics(step, console_metrics)
                    if wandb_run: wandb_run.log(logs, step=step)
                # Request a synchronized checkpoint+stop at an optimizer boundary.
                # Rank zero owns the file; the broadcast avoids filesystem-cache
                # differences causing ranks to save different training steps.
                if log_due:
                    request = output / "checkpoint_and_stop.request"
                    stop_requested = torch.tensor(
                        int(rank == 0 and request.exists()), device=device, dtype=torch.int32)
                    if world > 1:
                        dist.broadcast(stop_requested, src=0)
                    if bool(stop_requested.item()):
                        requested_stop = True
                        max_steps = step
                        if rank == 0:
                            request.unlink(missing_ok=True)
                            log_info(task="policy", phase="checkpoint_stop", step=step)
                window_loss.zero_(); window_metrics.clear()
                if save_every > 0 and step%save_every == 0:
                    _save_checkpoint(model, optimizer, scheduler, path=policy_weight_path(output, step),
                        rank=rank, world=world, fsdp=fsdp, codec_spec=codec_spec, contract=contract,
                        epoch=saved_epoch, next_batch=saved_next, step=step, config=cfg)
                performance.start_fetch()
            if saved_epoch == epoch+1 and val_ids:
                val_metrics = evaluate(model, dataset, val_ids, batch_size=batch_size, rank=rank, world=world,
                                       device=device, precision=precision, seed=data_seed+987, return_metrics=True)
                last_val = None if val_metrics is None else val_metrics["action_loss"]
                if rank == 0 and last_val is not None:
                    logs = {**{f"validation/{key}": value for key,value in val_metrics.items()}, "train/global_step": step}
                    with (output/"metrics.jsonl").open("a") as stream: stream.write(json.dumps(logs)+"\n")
                    log_info(task="policy", phase="eval", step=step, epoch=epoch+1, **val_metrics)
                    if wandb_run: wandb_run.log(logs, step=step)
            next_batch = 0
        log_training_performance(performance, output=output, device=device, rank=rank, world=world,
                                 step=step, epoch=max(1, saved_epoch+int(saved_next > 0)))
        final_checkpoint = policy_weight_path(output, step)
        _save_checkpoint(model, optimizer, scheduler, path=final_checkpoint, rank=rank, world=world, fsdp=fsdp,
            codec_spec=codec_spec, contract=contract, epoch=saved_epoch, next_batch=saved_next, step=step, config=cfg,
            save_training_state=requested_stop)
        log_info(task="policy", phase="complete", rank=rank, step=step, path=final_checkpoint)
    finally:
        progress.close()
    if rank == 0 and wandb_run: wandb_run.finish()
    return {"global_step": step, "checkpoint": str(policy_weight_path(output, step)), "validation_loss": last_val,
            "codec_spec": codec_spec, "distributed": "fsdp" if fsdp else "single"}



def load_config(path, _stack=()):
    """Resolve relative _base_ YAML inheritance with explicit cycle rejection."""
    path = Path(path).expanduser().resolve()
    if path in _stack:
        raise ValueError("Configuration inheritance cycle: " + " -> ".join(map(str, (*_stack, path))))
    current = OmegaConf.load(path)
    bases = current.pop("_base_", None)
    merged = OmegaConf.create({})
    for base in ([bases] if isinstance(bases, str) else list(bases or [])):
        merged = OmegaConf.merge(merged, load_config(path.parent / base, (*_stack, path)))
    return OmegaConf.merge(merged, current)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(load_config(args.config), OmegaConf.from_dotlist(overrides))
    if "trainer" not in cfg or cfg.trainer is None:
        cfg.trainer = {}
    trainer = cfg.trainer
    if trainer.get("max_checkpoints") is not None:
        raise ValueError("trainer.max_checkpoints is removed; every saved policy step is kept")
    config_path = Path(args.config).expanduser().resolve()
    stem = config_path.stem
    if not stem or stem.startswith("_"):
        raise ValueError(f"Policy config name cannot be a checkpoint directory: {config_path.name}")
    if not trainer.get("output_dir"):
        # Native config training defaults to RoboTwin. The benchmark launcher supplies
        # the exact run directory for both RoboTwin and EmbodiBench.
        seed = int(trainer.get("seed", 0))
        trainer.output_dir = str(ROOT / "checkpoints" / f"RoboTwin-{stem}-joint-{seed}")
    try: run_training(cfg)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    main()

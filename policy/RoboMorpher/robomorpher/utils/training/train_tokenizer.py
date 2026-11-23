"""Train the joint residual-VQ action tokenizer.

  - inputs (B, J, 7) / outputs (B, J, 10), no time dimension
  - lazy zarr dataloaders with rank-0-once normalization stats
  - gradient accumulation (gradient_accumulation_steps); cosine / wandb step on optimizer updates
  - FK diagnostics are evaluation-only; no FK term or auxiliary training decoder
"""

import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader
from accelerate.utils import GradientAccumulationPlugin, set_seed
from tqdm import tqdm, trange
from omegaconf import OmegaConf
import wandb
from robomorpher.utils.training.logging import log_info
from robomorpher.utils.training.state import (collect_tokenizer_training_state,
                                  restore_tokenizer_training_state, restore_rng_state)
from robomorpher.utils.training.checkpointing import (
    atomic_torch_save, prepare_output_directory, validate_checkpoint_destination,
    register_and_prune_checkpoint, resolve_max_checkpoints, assert_no_pending_ema,
)

from robomorpher.model.tokenizer import (
    build_tokenizer_from_config,
    normalize_model_config,
)
from robomorpher.model.tokenizer.utils.config import STEP_ARCHITECTURES, resolve_training_model_config
from robomorpher.utils.data.lazy_data.zarr_dataset import (
    create_dataloaders, coerce_norm_stats, dataset_norm_checkpoint,
)
from robomorpher.utils.tokenizer.scheduler import initialize_lr_schedule, lr_schedule_state
from robomorpher.utils.tokenizer.metrics import TokenizerMetricAccumulator
from robomorpher.utils.tokenizer.evaluation import compute_fk_ee_loss
from robomorpher.utils.tokenizer.reconstruction import (
    SPLIT_EE_LOSS, resolve_reconstruction_loss_config, output_norm_stats,
    masked_rotation_geodesic, is_split_ee_loss, masked_regression_loss,
)

os.environ["NCCL_P2P_DISABLE"] = "1"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


def get_config():
    """Parse: --config <yaml> plus optional overrides as --key value or key=value."""
    import sys

    args = sys.argv[1:]
    config_path = None
    overrides = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--config":
            config_path = args[i + 1]
            i += 2
        elif a.startswith("--") and "=" not in a and i + 1 < len(args) and not args[i + 1].startswith("--"):
            overrides.append(f"{a[2:]}={args[i + 1]}")
            i += 2
        else:
            overrides.append(a.lstrip("-") if a.startswith("--") else a)
            i += 1
    if config_path is None:
        raise ValueError("missing --config <yaml>")
    yaml_conf = OmegaConf.load(config_path)
    cli_conf = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(yaml_conf, cli_conf)


def build_lr_scheduler(optimizer, cfg, num_training_steps, last_epoch=-1):
    """Fresh-plan compatibility helper; checkpoint resume requires its recipe."""
    if last_epoch != -1:
        raise ValueError("Use initialize_lr_schedule with checkpoint metadata to resume a schedule")
    scheduler, _ = initialize_lr_schedule(optimizer, cfg.optim, num_training_steps)
    return scheduler


def resolve_ckpt_path(load_dir):
    if load_dir is None:
        return None
    path = str(load_dir).strip()
    if path == "" or path.lower() == "none":
        return None
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint(ckpt_path, accelerator, model, optimizer, ckpt=None):
    if ckpt is None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    accelerator.unwrap_model(model).load_state_dict(ckpt["model_state_dict"], strict=True)
    # Optimizer and LR schedule are restored together after the loader's update
    # count is known. Do not load old initial_lr and silently mix in a new lambda.

    start_epoch = int(ckpt.get("epoch", 0))
    global_step = ckpt.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("Checkpoint requires a nonnegative global_step; cannot infer successful updates from epoch")
    if accelerator.is_main_process:
        log_info(
            f"Loaded checkpoint: {ckpt_path} | "
            f"resume from epoch {start_epoch + 1} | global_step={global_step}"
        )
    return start_epoch, global_step


def extract_ckpt_norm(ckpt):
    """Return complete frozen stats and method; preserve schema metadata."""
    raw = ckpt.get("norm")
    if raw is None:
        return None, None
    norm_stats = coerce_norm_stats(raw)
    return norm_stats, norm_stats.get("normalization_method")


def save_tokenizer_checkpoint(payload, model, path, contract, max_checkpoints):
    """Preserve the established epoch payload; only I/O/retention are shared with VLA."""
    path = Path(path)
    assert_no_pending_ema(model)
    validate_checkpoint_destination(path.parent, path, contract, axis="epoch")
    atomic_torch_save(payload, path)
    register_and_prune_checkpoint(path.parent, path, contract, max_checkpoints,
                                  axis="epoch", allow_unlimited=True)
    log_info(phase="checkpoint", epoch=payload["epoch"], step=payload["global_step"], path=path)



def tokenizer_checkpoint_contract(cfg):
    """Stable save ownership across resumes; input checkpoint paths are not identity."""
    model_config = normalize_model_config(cfg.vqvae_model)
    model_config.pop("load_dir", None)
    contract = {
        "task": "tokenizer", "axis": "epoch", "version": 2,
        "model_config": model_config,
        "reconstruction_loss": resolve_reconstruction_loss_config(cfg.vqvae_model),
        "dataset": str(Path(cfg.train_data_dir).expanduser().resolve()),
        "normalization_method": cfg.get("normalization_method"),
        "normalization_scope": cfg.get("normalization_scope", "shared_feature"),
        "batch_size": int(cfg.batch_size),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
        "seed": int(cfg.seed),
    }
    angle_wrap = str(cfg.get("joint_angle_wrap", ""))
    if angle_wrap != "none":
        raise ValueError(
            "Tokenizer training requires joint_angle_wrap: none; "
            "principal-angle preprocessing has been removed; use raw physical radians."
        )
    contract["joint_angle_wrap"] = "none"
    return contract


def prepare_tokenizer_checkpoint_directory(cfg, checkpoint_path, checkpoint):
    """Use the exact save_path; protect old files and refuse branch rewrites."""
    output = Path(cfg.save_path).expanduser().resolve()
    contract = tokenizer_checkpoint_contract(cfg)
    if checkpoint_path and output.exists():
        resume_epoch = int(checkpoint["epoch"])
        later = [path.name for path in output.glob("epoch_*.pt")
                 if path.stem.removeprefix("epoch_").isdigit()
                 and int(path.stem.removeprefix("epoch_")) > resume_epoch]
        if later:
            raise ValueError(
                "Output directory contains epochs newer than the resume checkpoint: "
                f"{sorted(later)}; resume the latest checkpoint or choose a new save_path"
            )
    prepare_output_directory(output, checkpoint_path, contract, axis="epoch",
                             resume_counter=None if checkpoint is None else int(checkpoint["epoch"]))
    return output, contract


def _checkpoint_error(error):
    # Rank-zero filesystem failures must be visible to all ranks before the next forward.
    if torch.distributed.is_initialized():
        errors = [error]
        torch.distributed.broadcast_object_list(errors, src=0)
        error = errors[0]
    if error is not None:
        raise RuntimeError(error)


def build_model(cfg, state_dict=None):
    # shared with the policy side (utils/training/train_policy.py loads the same yaml)
    return build_tokenizer_from_config(cfg.vqvae_model, state_dict=state_dict)


def apply_optimizer_update(accelerator, optimizer, scheduler):
    """Update parameters and schedule; EMA is handled by the upstream forward.

    Accelerate makes optimizer.step()/zero_grad() no-ops during accumulation.
    Its overflow flag must be read after step(), not before. A skipped optimizer
    update does not roll back EMA already committed in the boundary forward.
    """
    optimizer.step()
    step_succeeded = accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped
    # The scheduler is deliberately not accelerator.prepare()'d, which would
    # multiply its updates by num_processes. Overflow must not advance it.
    if scheduler is not None and step_succeeded:
        scheduler.step()
    optimizer.zero_grad()
    return step_succeeded


def _masked_squared_error(pred, target, joint_valid_mask):
    """Clear padding before arithmetic, retaining at least FP32 loss precision."""
    valid = joint_valid_mask[..., None].to(device=pred.device, dtype=torch.bool)
    dtype = torch.promote_types(pred.dtype, target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    # Multiplying an already-computed NaN/Inf error by zero does not mask it,
    # and also leaves nonfinite gradients. Mask both operands before subtraction.
    clean_pred = pred.to(dtype=dtype).masked_fill(~valid, 0.0)
    clean_target = target.to(dtype=dtype).masked_fill(~valid, 0.0)
    return (clean_pred - clean_target).square(), valid


def masked_joint_mse(pred, target, joint_valid_mask):
    """MSE over valid joint elements only (T=1 shapes: pred (B, J, D), mask (B, J))."""
    squared_error, valid = _masked_squared_error(pred, target, joint_valid_mask)
    valid_elements = valid.sum(dtype=squared_error.dtype) * pred.shape[-1]
    return squared_error.sum() / valid_elements.clamp_min(1.0)


def per_joint_mse(pred, target, joint_valid_mask):
    """Per-joint masked MSE -> (num_joints,)."""
    squared_error, valid = _masked_squared_error(pred, target, joint_valid_mask)
    denom = (valid.sum(dim=(0, 2), dtype=squared_error.dtype) * pred.shape[-1]).clamp_min(1.0)
    return squared_error.sum(dim=(0, 2)) / denom


def compute_recon_losses(
    ee_pred, joint_pred, output_data, joint_valid_mask, ee_dim, joint_dim,
    *, loss_config=None, norm_stats=None,
):
    """Direct reconstruction with independent normalized xyz/joint L1 or MSE.

    defaults retain EE9 and joint MSE exactly. Split EE weights are
    applied here; the joint weight is applied by the caller exactly once.
    """
    # Output suppression never changes the dataset schema: pose9 precedes q.
    target_ee_dim = 9 if ee_dim == 0 else ee_dim
    if output_data.shape[-1] != target_ee_dim + joint_dim:
        raise ValueError("Target schema must contain pose channels followed by joint angles")
    ee_target = output_data[..., :ee_dim]
    joint_target = output_data[..., target_ee_dim : target_ee_dim + joint_dim]
    # Pure EE tokenizers intentionally have no joint-angle decoder channels.
    # Keep a differentiable zero joint objective and skip the D=0 regression
    # helper, which is defined only for non-empty feature dimensions.
    if joint_dim == 0:
        joint_loss = ee_pred.new_zeros(())
        angle_per_joint = ee_pred.new_zeros((ee_pred.shape[1],))
    else:
        joint_type = "l2" if loss_config is None else loss_config.get("joint_loss_type", "l2")
        if joint_type in {"l2", "mse"}:
            joint_loss = masked_joint_mse(joint_pred, joint_target, joint_valid_mask)
            angle_per_joint = per_joint_mse(joint_pred, joint_target, joint_valid_mask)
        else:
            joint_loss, angle_per_joint = masked_regression_loss(
                joint_pred, joint_target, joint_valid_mask, joint_type,
            )
    if ee_dim == 0:
        if ee_pred.shape != (*joint_pred.shape[:2], 0):
            raise ValueError("angle-only decoding requires empty EE predictions")
        return joint_loss.new_zeros(()), joint_loss, {"angle": angle_per_joint}
    # Keep the historical split name valid for checkpoint/probe callers.
    if loss_config is not None and loss_config["ee_type"] in {SPLIT_EE_LOSS, "xyz_rot_geodesic"}:
        if ee_dim != 9:
            raise ValueError("Split EE loss requires xyz + rot6D")
        xyz_type = loss_config.get("xyz_loss_type", "l2")
        if xyz_type in {"l2", "mse"}:
            xyz_loss = masked_joint_mse(ee_pred[..., :3], ee_target[..., :3], joint_valid_mask)
            xyz_per_joint = per_joint_mse(ee_pred[..., :3], ee_target[..., :3], joint_valid_mask)
        else:
            xyz_loss, xyz_per_joint = masked_regression_loss(
                ee_pred[..., :3], ee_target[..., :3], joint_valid_mask, xyz_type,
            )
        rot_loss, rot_per_joint = masked_rotation_geodesic(
            ee_pred, ee_target, joint_valid_mask, norm_stats,
        )
        wx, wr = loss_config["xyz_weight"], loss_config["rot_weight"]
        return wx * xyz_loss + wr * rot_loss, joint_loss, {
            "ee": wx * xyz_per_joint + wr * rot_per_joint,
            "angle": angle_per_joint, "xyz": xyz_per_joint, "rotation": rot_per_joint,
            "pos_loss": xyz_loss.detach(), "rot_loss": rot_loss.detach(),
        }
    ee_loss = masked_joint_mse(ee_pred, ee_target, joint_valid_mask)
    per_joint = {
        "ee": per_joint_mse(ee_pred, ee_target, joint_valid_mask),
        "angle": angle_per_joint,
    }
    # configs retain their original training objective. Expose its
    # normalized coordinate components for the same compact log schema.
    if ee_dim == 9:
        with torch.no_grad():
            per_joint["pos_loss"] = masked_joint_mse(ee_pred[..., :3], ee_target[..., :3], joint_valid_mask)
            per_joint["rot_loss"] = masked_joint_mse(ee_pred[..., 3:9], ee_target[..., 3:9], joint_valid_mask)
    return ee_loss, joint_loss, per_joint


def joint_features_ema_count_reference(train_dataset, max_num_joints):
    """Expected valid vectors per frame from the verified training-only stats.

    This sets one fixed count-scale reference, not a per-batch threshold.
    Actual k-means, EMA, commitment, and usage still consume the true mask.
    """
    metadata = getattr(train_dataset, "norm_metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("joint_features EMA threshold requires versioned training normalization metadata")
    split = metadata.get("train_split")
    if (metadata.get("valid_rule") != "joint_valid_mask_true_training_rows_only"
            or metadata.get("max_num_joints") != int(max_num_joints)
            or not isinstance(split, dict)):
        raise ValueError("joint_features EMA count metadata must describe the configured valid training joints")
    train_rows, valid_tokens = split.get("train_rows"), metadata.get("valid_train_tokens")
    if (isinstance(train_rows, bool) or not isinstance(train_rows, int) or train_rows <= 0
            or isinstance(valid_tokens, bool) or not isinstance(valid_tokens, int)
            or not train_rows <= valid_tokens <= train_rows * int(max_num_joints)):
        raise ValueError("joint_features EMA threshold requires positive train_rows and valid_train_tokens")
    if getattr(train_dataset, "split", None) != "train" or len(train_dataset) != train_rows:
        raise ValueError("joint_features EMA count reference must match the actual training split")
    return {
        "source": "training_normalization_valid_joint_counts",
        "valid_train_tokens": valid_tokens,
        "train_rows": train_rows,
        "mean_valid_joints_per_frame": valid_tokens / train_rows,
        "train_split": dict(split),
    }


def resolve_frame_ema_threshold(cfg, accelerator, grad_accum, checkpoint=None, train_dataset=None):
    """Set a fixed frame-RVQ threshold in the upstream EMA count units.

    Counts are summed over valid vectors, ranks, and accumulation microbatches.
    Learned queries retain their original fixed-slot reference. Joint features
    use the training-set mean valid-joint count, excluding padding. Both are
    nominal full-batch expectations; neither changes the threshold each step.
    Preserve the checkpoint threshold on resume. An explicit fresh threshold
    overrides the fractional default, but joint-mode metadata is still audited.
    """
    vm = normalize_model_config(cfg.vqvae_model)
    if vm["architecture"] not in STEP_ARCHITECTURES:
        return
    ema = cfg.vqvae_model.quantizer.ema
    count_accum = int(grad_accum) if vm["ema_update_unit"] == "accumulation_boundary" else 1
    joint_features = vm.get("frame_latent_mode", "learn_query") == "joint_features"
    count_reference = None
    if joint_features:
        count_reference = joint_features_ema_count_reference(train_dataset, vm["max_num_joints"])
        vectors_per_frame = count_reference["mean_valid_joints_per_frame"]
    else:
        vectors_per_frame = int(vm["frame_num_latents"])
    effective_vectors = (
        int(cfg.batch_size) * int(accelerator.num_processes)
        * count_accum * vectors_per_frame
    )
    explicit_threshold = "dead_code_threshold" in ema
    saved_ema = {}
    if checkpoint is not None:
        saved_cfg = checkpoint.get("model_config") or {}
        saved_meta = checkpoint.get("quantizer_metadata") or {}
        saved_vm = normalize_model_config(saved_cfg)
        threshold = saved_meta.get("dead_code_threshold", saved_vm["dead_code_threshold"])
        saved_ema = (saved_cfg.get("quantizer") or {}).get("ema") or {}
        # Any different explicit value on resume is an error rather than an
        # unnoticed codebook-update policy change.
        if "dead_code_threshold" in ema and float(ema.dead_code_threshold) != float(threshold):
            raise ValueError("Resume dead-code threshold differs from checkpoint; use a fresh run")
        ema.dead_code_threshold = float(threshold)
    elif "dead_code_threshold" not in ema:
        fraction = float(vm["dead_code_fraction"])
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("dead_code_fraction must be between zero and one")
        ema.dead_code_threshold = fraction * effective_vectors / int(vm["codebook_size"])
    ema.nominal_vectors_per_update = effective_vectors
    ema.threshold_policy = "checkpoint_preserved" if checkpoint is not None else (
        "explicit" if explicit_threshold else (
            "fraction_of_expected_valid_joint_occupancy" if joint_features
            else "fraction_of_nominal_uniform_occupancy"
        )
    )
    if joint_features:
        ema.count_reference = count_reference
        # A changed batch/world/accum on resume does not redefine the threshold.
        # Keep its original reference separate from today's expected vector count.
        ema.threshold_reference_vectors_per_update = (
            saved_ema.get("threshold_reference_vectors_per_update",
                          saved_ema.get("nominal_vectors_per_update"))
            if checkpoint is not None else effective_vectors
        )


def training_quantizer_metadata(model, cfg):
    metadata = model.quantizer_metadata()
    vm = normalize_model_config(cfg.vqvae_model)
    if vm.get("frame_latent_mode", "learn_query") == "joint_features":
        ema = cfg.vqvae_model.quantizer.ema
        metadata.update(
            nominal_vectors_per_update=float(ema.nominal_vectors_per_update),
            dead_code_threshold_policy=str(ema.threshold_policy),
            ema_count_reference=OmegaConf.to_container(ema.count_reference, resolve=True),
            threshold_reference_vectors_per_update=ema.threshold_reference_vectors_per_update,
        )
    return metadata


def accelerator_precision_kwargs(cfg):
    """Honor explicit YAML/CLI precision without changing environment defaults."""
    configured = cfg.get("mixed_precision", None)
    if configured is None:
        return {}
    # OmegaConf/YAML parses the unquoted CLI value no as boolean False.
    precision = "no" if configured is False else str(configured).lower()
    if precision not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be one of: no, fp16, bf16")
    return {"mixed_precision": precision}


def prepare_evaluation_loader(data_loader, accelerator):
    """Shard evaluation records exactly once, without padded repeat samples.

    Evaluation can have uneven/empty ranks, so this loader does not dispatch
    batches or synchronize RNG states. Its sequential sampler needs neither.
    Use an unwrapped eval model and reduce diagnostics only after every rank
    has finished its local loop. Training loader/Accelerator policy is unchanged.
    """
    return prepare_data_loader(
        data_loader,
        accelerator.device,
        num_processes=accelerator.num_processes,
        process_index=accelerator.process_index,
        split_batches=accelerator.split_batches,
        put_on_device=True,
        rng_types=[],
        dispatch_batches=False,
        even_batches=False,
        non_blocking=accelerator.non_blocking,
    )


def create_training_dataloaders(cfg, accelerator, ckpt):
    # Resume: freeze normalization from the checkpoint (not a possibly-recomputed npz).
    resume_norm_stats = None
    resume_norm_method = None
    if ckpt is not None:
        resume_norm_stats, resume_norm_method = extract_ckpt_norm(ckpt)
        if resume_norm_stats is None:
            raise ValueError(
                "Training resume requires frozen checkpoint normalization. This checkpoint has no norm; "
                "refusing to replace its training statistics with a new cache. Historical evaluation "
                "may explicitly use its original cache; a fresh normalization requires a new run."
            )
        elif accelerator.is_main_process:
            log_info(
                f"[stats] resume norm from checkpoint "
                f"(method={resume_norm_method or cfg.get('normalization_method')})"
            )

    return create_dataloaders(
        cfg,
        accelerator,
        norm_stats=resume_norm_stats,
        normalization_method=resume_norm_method,
    )



def main(cfg):
    grad_accum = max(int(cfg.get("gradient_accumulation_steps", 1)), 1)
    # Do not force a sync on the last leftover micro-batches: backward() always
    # divides by accum, so a 3-batch tail would be a 3/8-strength update.
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=grad_accum,
            sync_with_dataloader=False,
        ),
        **accelerator_precision_kwargs(cfg),
    )
    device = accelerator.device

    set_seed(cfg.seed)
    seed_everything(cfg.seed)

    # Resolve the checkpoint format before constructing the quantizer. Old
    # EMA states remain available for strict evaluation, but this
    # training entry point must not silently continue the unrepaired backend.
    vm = normalize_model_config(cfg.vqvae_model)
    ckpt_path = resolve_ckpt_path(vm.get("load_dir"))
    ckpt = None if ckpt_path is None else torch.load(ckpt_path, map_location="cpu")
    cfg.vqvae_model = OmegaConf.create(resolve_training_model_config(cfg.vqvae_model, ckpt))
    vm = normalize_model_config(cfg.vqvae_model)
    prepared_loaders = None
    if vm.get("frame_latent_mode", "learn_query") == "joint_features":
        # Only this new mode needs training-mask statistics before construction.
        # The learned-query path keeps its historical initialization order.
        prepared_loaders = create_training_dataloaders(cfg, accelerator, ckpt)
    resolve_frame_ema_threshold(
        cfg, accelerator, grad_accum, checkpoint=ckpt,
        train_dataset=None if prepared_loaders is None else prepared_loaders[0],
    )
    vm = normalize_model_config(cfg.vqvae_model)
    loss_config = resolve_reconstruction_loss_config(cfg.vqvae_model)
    model = build_model(cfg, state_dict=None if ckpt is None else ckpt["model_state_dict"])
    quantizer_metadata = training_quantizer_metadata(model, cfg)
    if ckpt is not None and accelerator.is_main_process and vm["use_ema_quantizer"]:
        saved_unit = ((ckpt.get("quantizer_metadata") or {}).get("ema_update_unit")
                      or normalize_model_config(ckpt.get("model_config") or {})["ema_update_unit"])
        if str(saved_unit).lower() != quantizer_metadata["ema_update_unit"]:
            log_info(
                f"[quantizer] Explicit EMA policy change: {saved_unit} -> "
                f"{quantizer_metadata['ema_update_unit']}; this is not an identical training resume."
            )
    if accelerator.is_main_process:
        # Credentials may come from the process environment / wandb login.
        # Never require or persist an API key in the new YAML/checkpoint config.
        if cfg.wandb.get("api_key"):
            os.environ["WANDB_API_KEY"] = str(cfg.wandb.api_key)
        public_config = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(public_config.get("wandb"), dict):
            public_config["wandb"].pop("api_key", None)
        # W&B creates its own wandb/ subdirectory inside this run's checkpoint directory.
        wandb_dir = Path(cfg.wandb.get("dir") or cfg.save_path).expanduser().resolve()
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=str(wandb_dir),
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            name=cfg.wandb.get("name", f"tok-{datetime.now().strftime('%m%d_%H%M')}"),
            config=public_config,
            mode=cfg.wandb.get("mode", None),
        )
        os.makedirs(cfg.save_path, exist_ok=True)
        log_info(f"[quantizer] {quantizer_metadata}")
        n_params = sum(p.numel() for p in model.parameters())
        log_info(f"Model params: {n_params / 1e6:.2f}M")
        if getattr(model, "use_urdf_split", False):
            log_info(
                f"[encoder] urdf split: urdf_layers={model.urdf_layers} "
                f"angle_layers={model.angle_layers} state_layers={model.state_layers} "
                f"fusion={model.urdf_angle_fusion} urdf_dim={model.urdf_dim}"
            )

    if prepared_loaders is None:
        prepared_loaders = create_training_dataloaders(cfg, accelerator, ckpt)
    train_dataset, test_dataset, train_loader, test_loader = prepared_loaders

    betas = cfg.optim.get("betas", [0.9, 0.999])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.get("weight_decay", 0.0),
        betas=tuple(float(b) for b in betas),
    )

    # Keep the original custom sampler: Accelerate versions do not all forward
    # set_epoch through BatchSamplerShard to a custom batch sampler.
    train_batch_sampler = train_loader.batch_sampler
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    test_loader = prepare_evaluation_loader(test_loader, accelerator)
    accumulate_vq_ema = quantizer_metadata["ema_update_unit"] == "accumulation_boundary"

    start_epoch = 0
    global_step = 0
    if ckpt_path is not None:
        start_epoch, global_step = load_checkpoint(
            ckpt_path, accelerator, model, optimizer, ckpt=ckpt
        )
    accelerator.wait_for_everyone()

    micro_steps_per_epoch = max(len(train_loader), 1)
    updates_per_epoch = micro_steps_per_epoch // grad_accum
    if updates_per_epoch < 1:
        raise ValueError(
            f"train loader has {micro_steps_per_epoch} micro-batches; "
            f"need at least gradient_accumulation_steps={grad_accum}"
        )
    micro_steps_used = updates_per_epoch * grad_accum
    total_steps = int(cfg.epochs) * updates_per_epoch
    cosine_epochs = cfg.optim.get("cosine_epochs", None)
    sched_steps = (
        int(cosine_epochs) * updates_per_epoch if cosine_epochs is not None else total_steps
    )
    scheduler, lr_recipe = initialize_lr_schedule(
        optimizer, cfg.optim, sched_steps, checkpoint=ckpt, global_step=global_step
    )
    if accelerator.is_main_process:
        eff_batch = int(cfg.batch_size) * accelerator.num_processes * grad_accum
        log_info(
            f"[optim] policy={lr_recipe['optimizer_resume_policy']} "
            f"actual_next_lr={[group['lr'] for group in optimizer.param_groups]} "
            f"base_lrs={lr_recipe['base_lrs']} min_lrs={lr_recipe['min_lrs']} "
            f"scheduler={lr_recipe['scheduler']} warmup={lr_recipe['warmup_steps']} "
            f"schedule_steps={lr_recipe['num_training_steps']} "
            f"schedule_start_global_step={lr_recipe['start_global_step']} "
            f"schedule_completed_updates={lr_recipe['completed_updates']} "
            f"train_steps={total_steps} accum={grad_accum} "
            f"micro/epoch={micro_steps_used}/{micro_steps_per_epoch} "
            f"updates/epoch={updates_per_epoch} effective_batch={eff_batch}"
        )
        if ckpt is not None:
            log_info(
                "[optim] Checkpoint optimizer state is retained (momentum, betas, weight decay). "
                + ("strict_continue: saved LR recipe takes precedence over current YAML schedule settings."
                   if lr_recipe["optimizer_resume_policy"] == "preserve_checkpoint_optimizer_and_lr_clock"
                   else "reset: current YAML defines a new LR plan starting now; old schedule state is ignored.")
            )
        leftover = micro_steps_per_epoch - micro_steps_used
        if leftover:
            log_info(
                f"[optim] drop last {leftover} micro-batches/epoch "
                f"(not a multiple of accum={grad_accum})"
            )

    eval_every_epoch = int(cfg.get("eval_every_epoch", 1))
    max_checkpoints = resolve_max_checkpoints(cfg, 8, allow_unlimited=True)
    log_every = int(cfg.get("log_every", 50))
    if log_every < 1:
        raise ValueError("log_every must be >= 1")
    output_with_ee = bool(vm.get("output_with_ee", True))
    ee_dim = int(vm["decoder_ee_dim"]) if output_with_ee else 0
    joint_dim = int(vm["decoder_joint_dim"])
    input_with_ee = bool(vm.get("input_with_ee", False))
    entropy_loss_weight = float(vm["entropy_loss_weight"])
    ee_loss_weight = loss_config["ee_weight"]
    joint_loss_weight = loss_config["joint_weight"]
    split_ee_loss = output_with_ee and is_split_ee_loss(loss_config["ee_type"])
    recon_norm_stats = output_norm_stats(train_dataset, device) if split_ee_loss else None
    ae_warmup_epochs = int(cfg.vqvae_model.quantizer.get("ae_warmup_epochs", 0))
    if ae_warmup_epochs < 0:
        raise ValueError("quantizer.ae_warmup_epochs must be nonnegative")
    if ae_warmup_epochs and vm["architecture"] not in STEP_ARCHITECTURES:
        raise ValueError("AE warm-up is currently supported only by action-step RVQ architectures")
    eval_fk_enabled = bool(cfg.get("evaluation", {}).get("fk", {}).get("enable", True))
    if accelerator.is_main_process:
        log_info(f"[angle] wrap={train_dataset.angle_wrap}; "
                 f"physical radians preserved={train_dataset.angle_wrap == 'none'}; "
                 f"frozen input range=[{train_dataset.x_min[..., -1].min().item():.8f}, "
                 f"{train_dataset.x_max[..., -1].max().item():.8f}] rad")
    num_joints = int(vm["max_num_joints"])
    metric_metadata = {
        "schema_version": 12,
        "output_with_ee": output_with_ee,
        "target_schema": "xyz3_rot6d6_angle1",
        "reconstruction_loss": loss_config,
        "ee_loss_definition": (
            f"xyz_weight * normalized_xyz_{'mae' if loss_config['xyz_loss_type'] == 'l1' else 'mse'}"
            " + rot_weight * physical_theta_over_pi"
            if split_ee_loss else ("normalized_ee9_mse" if output_with_ee else "disabled_no_ee_output")
        ),
        "angle_loss_definition": (
            "normalized_joint_mae" if loss_config["joint_loss_type"] == "l1"
            else "normalized_joint_mse"
        ),
        "l2_definition": "mean_squared_error_not_unsquared_euclidean_norm",
        "rotation_acos_eps": 1e-7 if split_ee_loss else None,
        "step_scope": "all_ranks_complete_accumulation_window",
        "joint_reduction": "valid_count_weighted_per_joint_objective",
        "scalar_reduction": "rank_microbatch_mean_of_training_objective",
        "empty_joint_loss": "omit_loss_when_no_valid_observations",
        "wandb_metrics": (["pos_loss", "angle_loss", "rot_loss", "total_loss", "code_usage"]
                          if output_with_ee else ["angle_loss", "total_loss", "code_usage"]),
        "wandb_step": "successful_optimizer_update",
        "wandb_component_weights": "unweighted_losses",
        "wandb_rot_loss": ("physical_theta_over_pi" if split_ee_loss else "normalized_rot6d_mse")
                          if output_with_ee else None,
        "wandb_total_loss": "actual_backward_objective_including_vq_and_weighted_entropy_excluding_fk",
        "wandb_code_usage": "mean_stage_usage_percentage_0_to_100_then_rank_microbatch_mean_not_window_union",
        "wandb_code_usage_during_ae": "0 means quantizer bypassed, not codebook collapse",
        "wandb_epoch_metrics": False,
        "wandb_per_joint_metrics": True,
        "wandb_per_joint_names": (["pretrain/pos_loss_joint_{j}", "pretrain/rot_loss_joint_{j}",
                                  "pretrain/angle_loss_joint_{j}"] if output_with_ee
                                  else ["pretrain/angle_loss_joint_{j}"]),
        "wandb_per_joint_record_order": ("pos_then_rot_then_angle_each_in_ascending_joint_index"
                                        if output_with_ee else "angle_in_ascending_joint_index"),
        "wandb_learning_rate": True,
        "wandb_learning_rate_names": ["pretrain/learning_rate", "pretrain/learning_rate_group_{i}"],
        "wandb_learning_rate_scope": "actual_lr_used_before_optimizer_step_not_next_scheduler_lr",
        "wandb_test_metrics": False,
        "epoch_includes_skipped_optimizer_windows": True,
        "sample_scope": "records_supplied_by_prepared_dataloader",
        "eval_sample_scope": "original_test_records_once_without_repeat_padding",
        "eval_forward": "unwrapped_frozen_model_no_loop_collectives",
        "joint_angle_wrap": train_dataset.angle_wrap,
        "angle_reconstruction_target": "raw_radians",
        "fk_scope": "evaluation_only",
        "eval_fk_enabled": eval_fk_enabled,
        "total_loss_excludes_fk": True,
        "ae_warmup_epochs": ae_warmup_epochs,
    }
    if accelerator.is_main_process:
        log_info(f"[metrics] {metric_metadata}")
        log_info(
            f"[loss] {loss_config}; "
            f"AE warm-up={ae_warmup_epochs} epochs; FK is evaluation-only"
        )
        if ckpt is not None:
            saved_loss_config = resolve_reconstruction_loss_config(ckpt.get("model_config") or {})
            if saved_loss_config != loss_config:
                log_info(f"[loss] Explicit objective change on resume: {saved_loss_config} -> {loss_config}; "
                      "new pos/angle/EE/total curves may not be comparable to the old objective")
        log_info(
            f"[arch] dual_branch_decoder={vm['dual_branch_decoder']}, "
            f"ema_quantizer={vm['use_ema_quantizer']}, rotation_trick={vm['use_rotation_trick']}, "
            f"encoder.input_with_ee={input_with_ee}"
        )

    eval_norm_stats = None
    if eval_fk_enabled:
        eval_norm_stats = {"method": train_dataset.normalization_method}
        stat_keys = ["x_mean", "x_std", "y_mean", "y_std", "x_min", "x_max", "y_min", "y_max"]
        if getattr(train_dataset, "quantile", None) is not None:
            stat_keys += ["x_q01", "x_q99", "y_q01", "y_q99"]
        for key in stat_keys:
            eval_norm_stats[key] = getattr(train_dataset, key).to(device)

    # The configured directory is the run directory; no hidden date level.
    save_path = Path(cfg.save_path).expanduser().resolve()
    checkpoint_contract = tokenizer_checkpoint_contract(cfg)
    error = None
    if accelerator.is_main_process:
        try:
            prepare_tokenizer_checkpoint_directory(cfg, ckpt_path, ckpt)
            log_info(phase="setup", epoch=f"{start_epoch}/{cfg.epochs}", step=global_step,
                     save_every_epoch=cfg.save_every_epoch, max_checkpoints=max_checkpoints,
                     output_dir=save_path)
        except Exception as exc:
            error = f"Checkpoint directory setup failed: {exc}"
    _checkpoint_error(error)
    accelerator.wait_for_everyone()

    if ckpt is not None:
        exact_state = restore_tokenizer_training_state(ckpt, accelerator)
        if not exact_state and accelerator.is_main_process:
            log_info(phase="resume", warning="checkpoint has no RNG/GradScaler state; "
                     "weights/optimizer/LR restored, but stochastic continuation is not exact")

    for epoch in trange(
        start_epoch,
        cfg.epochs,
        initial=start_epoch,
        total=cfg.epochs,
        disable=not accelerator.is_main_process or not sys.stderr.isatty(),
    ):
        train_batch_sampler.set_epoch(epoch)
        # Restore the prepared loader's own iteration clock as well. Otherwise
        # a newly constructed loader can reset a resumed sampler to epoch zero.
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        model.train()
        train_metrics = TokenizerMetricAccumulator(
            num_joints, ee_dim, joint_dim, device, ee_loss_type=loss_config["ee_type"],
            record_components=(ee_dim == 9),
            xyz_loss_type=loss_config["xyz_loss_type"],
            joint_loss_type=loss_config["joint_loss_type"],
        )
        window_metrics = TokenizerMetricAccumulator(
            num_joints, ee_dim, joint_dim, device, ee_loss_type=loss_config["ee_type"],
            record_components=(ee_dim == 9),
            xyz_loss_type=loss_config["xyz_loss_type"],
            joint_loss_type=loss_config["joint_loss_type"],
        )

        for micro_i, (input_data, output_data, joint_valid_mask) in enumerate(
            tqdm(
                train_loader,
                disable=not accelerator.is_main_process or not sys.stderr.isatty(),
                dynamic_ncols=True,
                leave=False,
            )
        ):
            if micro_i >= micro_steps_used:
                break
            input_data = input_data.to(device, non_blocking=True)
            output_data = output_data.to(device, non_blocking=True)
            joint_valid_mask = joint_valid_mask.to(device, non_blocking=True).bool()

            with accelerator.accumulate(model):
                # Accelerate derives this boundary from cfg.gradient_accumulation_steps.
                # The final microbatch invokes the upstream non-accumulating forward.
                vq_kwargs = ({"accum_ema_update": not accelerator.sync_gradients}
                             if accumulate_vq_ema else {})
                if ae_warmup_epochs and epoch < ae_warmup_epochs:
                    # First learn a reconstructive continuous latent.  The first
                    # post-warm-up forward initializes EMA k-means from that
                    # trained latent distribution instead of a drifting random encoder.
                    vq_kwargs = {"bypass_quantizer": True}
                if input_with_ee:
                    # The same normalized all-joint EE target is an encoder
                    # input only. Keep x/URDF normalization and decoder inputs unchanged.
                    vq_kwargs["ee_features"] = output_data[..., :9]
                ee_pred, joint_pred, vq_loss, entropy_loss, usages = model(
                    input_data, joint_valid_mask, **vq_kwargs,
                )
                ee_loss, joint_loss, per_joint = compute_recon_losses(
                    ee_pred, joint_pred, output_data, joint_valid_mask, ee_dim, joint_dim,
                    loss_config=loss_config, norm_stats=recon_norm_stats,
                )
                recon_loss = ee_loss_weight * ee_loss + joint_loss_weight * joint_loss
                loss = (
                    recon_loss
                    + vq_loss
                    + entropy_loss_weight * entropy_loss
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer_lrs_used = [group["lr"] for group in optimizer.param_groups]
                step_succeeded = apply_optimizer_update(
                    accelerator, optimizer, scheduler
                )

            # Diagnostics only: detached sufficient statistics do not change
            # the loss, gradients, optimizer clock, or upstream EMA ordering.
            batch_metrics = window_metrics.update(
                ee_pred, joint_pred, output_data, joint_valid_mask,
                ee_per_joint=per_joint["ee"] if split_ee_loss else None,
                rot_per_joint=per_joint["rotation"] if split_ee_loss else None,
                scalars={
                    "total_loss": loss, "ee_loss": ee_loss, "joint_loss": joint_loss,
                    "recon_loss": recon_loss, "vq_loss": vq_loss,
                    "entropy_loss": entropy_loss,
                    "code_usages": usages,
                    **({"pos_loss": per_joint["pos_loss"], "rot_loss": per_joint["rot_loss"]}
                       if ee_dim == 9 else {}),
                },
            )
            train_metrics.add(batch_metrics)

            # All ranks participate even when AMP skips the parameter update.
            # Clear every completed window to prevent statistics leaking from
            # a skipped update into the next successful step's log.
            global_window_metrics = None
            if accelerator.sync_gradients:
                global_window_metrics = window_metrics.synchronize(accelerator)
                window_metrics.reset()

            if step_succeeded:
                if accelerator.is_main_process:
                    public_metrics = window_metrics.training_wandb_dict(global_window_metrics)
                    public_metrics.update(window_metrics.training_per_joint_dict("pretrain", global_window_metrics))
                    public_metrics["pretrain/learning_rate"] = float(optimizer_lrs_used[0])
                    for group_index, group_lr in enumerate(optimizer_lrs_used[1:], start=1):
                        public_metrics[f"pretrain/learning_rate_group_{group_index}"] = float(group_lr)
                    if public_metrics:
                        wandb.log(public_metrics, step=global_step)
                    if (global_step + 1) % log_every == 0 or global_step == 0:
                        scalars = window_metrics.scalar_means(global_window_metrics)
                        log_info(phase="train", epoch=f"{epoch + 1}/{cfg.epochs}", step=global_step + 1,
                                 total_loss=scalars["total_loss"], pos_loss=scalars.get("pos_loss"),
                                 rot_loss=scalars.get("rot_loss"), angle_loss=scalars["joint_loss"],
                                 vq_loss=scalars["vq_loss"], code_usage=scalars["code_usages"],
                                 lr=optimizer_lrs_used, grad_norm=float(grad_norm))
                global_step += 1

        train_metric_stats = train_metrics.synchronize(accelerator)
        train_scalar_means = train_metrics.scalar_means(train_metric_stats)
        mean_train_loss = train_scalar_means["total_loss"]
        mean_train_ee_loss = train_scalar_means["ee_loss"]
        mean_train_joint_loss = train_scalar_means["joint_loss"]
        mean_train_entropy_loss = train_scalar_means["entropy_loss"]

        need_eval = ((epoch + 1) % eval_every_epoch == 0) or (epoch + 1 == cfg.epochs)
        mean_test_loss = None
        mean_test_ee_loss = None
        mean_test_joint_loss = None
        mean_test_entropy_loss = None
        mean_test_fk_loss = None
        if need_eval:
            model.eval()
            # DDP forward broadcasts buffers; uneven evaluation ranks must not
            # call it. unwrap_model retains Accelerate's autocast/FP32 wrapper.
            eval_model = accelerator.unwrap_model(model)
            test_metrics = TokenizerMetricAccumulator(
                num_joints, ee_dim, joint_dim, device, include_fk=eval_fk_enabled,
                ee_loss_type=loss_config["ee_type"], record_components=(ee_dim == 9),
                xyz_loss_type=loss_config["xyz_loss_type"],
                joint_loss_type=loss_config["joint_loss_type"],
            )
            # Some Accelerate versions yield a None sentinel for an empty
            # prepared loader; do not try to unpack a batch when length is zero.
            eval_batches = test_loader if len(test_loader) > 0 else ()
            with torch.no_grad():
                for input_data, output_data, joint_valid_mask in eval_batches:
                    input_data = input_data.to(device, non_blocking=True)
                    output_data = output_data.to(device, non_blocking=True)
                    joint_valid_mask = joint_valid_mask.to(device, non_blocking=True).bool()

                    eval_kwargs = ({"bypass_quantizer": True}
                                   if ae_warmup_epochs and epoch < ae_warmup_epochs else {})
                    if input_with_ee:
                        eval_kwargs["ee_features"] = output_data[..., :9]
                    ee_pred, joint_pred, vq_loss, entropy_loss, usages = eval_model(
                        input_data, joint_valid_mask, **eval_kwargs,
                    )
                    ee_loss, joint_loss, per_joint = compute_recon_losses(
                        ee_pred, joint_pred, output_data, joint_valid_mask, ee_dim, joint_dim,
                        loss_config=loss_config, norm_stats=recon_norm_stats,
                    )
                    recon_loss = ee_loss_weight * ee_loss + joint_loss_weight * joint_loss
                    fk_loss, fk_per_joint = None, None
                    if eval_fk_enabled:
                        fk_loss, fk_per_joint = compute_fk_ee_loss(
                            input_data, output_data, joint_pred, joint_valid_mask,
                            eval_norm_stats, ee_dim, joint_dim,
                        )
                    loss = (
                        recon_loss
                        + vq_loss
                        + entropy_loss_weight * entropy_loss
                    )
                    test_metrics.update(
                        ee_pred, joint_pred, output_data, joint_valid_mask,
                        fk_per_joint=fk_per_joint,
                        ee_per_joint=per_joint["ee"] if split_ee_loss else None,
                        rot_per_joint=per_joint["rotation"] if split_ee_loss else None,
                        scalars={
                            "total_loss": loss, "ee_loss": ee_loss, "joint_loss": joint_loss,
                            "recon_loss": recon_loss, "vq_loss": vq_loss,
                            "entropy_loss": entropy_loss,
                            **({"fk_loss": fk_loss} if eval_fk_enabled else {}),
                            "code_usages": usages,
                            **({"pos_loss": per_joint["pos_loss"], "rot_loss": per_joint["rot_loss"]}
                               if ee_dim == 9 else {}),
                        },
                    )

            test_metric_stats = test_metrics.synchronize(accelerator)
            test_scalar_means = test_metrics.scalar_means(test_metric_stats)
            # test_split=0 is supported. An empty validation set has no loss;
            # retain None instead of logging zero or indexing an empty summary.
            if test_scalar_means:
                mean_test_loss = test_scalar_means["total_loss"]
                mean_test_ee_loss = test_scalar_means["ee_loss"]
                mean_test_joint_loss = test_scalar_means["joint_loss"]
                mean_test_entropy_loss = test_scalar_means["entropy_loss"]
                mean_test_fk_loss = test_scalar_means.get("fk_loss")

        if accelerator.is_main_process:
            log_info(phase="epoch", epoch=f"{epoch + 1}/{cfg.epochs}", step=global_step,
                     total_loss=mean_train_loss, pos_loss=train_scalar_means.get("pos_loss"),
                     rot_loss=train_scalar_means.get("rot_loss"), angle_loss=mean_train_joint_loss,
                     vq_loss=train_scalar_means["vq_loss"], code_usage=train_scalar_means["code_usages"],
                     lr=optimizer_lrs_used)
            if mean_test_loss is not None:
                log_info(phase="eval", epoch=f"{epoch + 1}/{cfg.epochs}", step=global_step,
                         total_loss=mean_test_loss, pos_loss=test_scalar_means.get("pos_loss"),
                         rot_loss=test_scalar_means.get("rot_loss"), angle_loss=mean_test_joint_loss,
                         fk_loss=mean_test_fk_loss)

        if (epoch + 1) % cfg.save_every_epoch == 0 or epoch + 1 == cfg.epochs:
            # All ranks collect before rank-zero serialization. Restore this RNG
            # after saving so checkpoint IO never changes subsequent training.
            training_state, entry_rng = collect_tokenizer_training_state(accelerator)
            error = None
            if accelerator.is_main_process:
                try:
                    quantizer_metadata = training_quantizer_metadata(accelerator.unwrap_model(model), cfg)
                    quantizer_metadata["gradient_accumulation_steps"] = grad_accum
                    checkpoint_model_config = OmegaConf.to_container(cfg.vqvae_model, resolve=True)
                    checkpoint_model_config["quantizer_backend"] = quantizer_metadata["backend"]
                    if isinstance(checkpoint_model_config.get("quantizer"), dict):
                        checkpoint_model_config["quantizer"]["backend"] = quantizer_metadata["backend"]
                    checkpoint_norm = dataset_norm_checkpoint(train_dataset)
                    ckpt = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "training_state": training_state,
                        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
                        "lr_schedule": lr_schedule_state(optimizer, scheduler, lr_recipe, global_step),
                        "train_loss": mean_train_loss,
                        "test_loss": mean_test_loss,
                        "norm": checkpoint_norm,
                        # tokenizer architecture travels with the checkpoint so the VLA
                        # side only needs model_path (no duplicated arch params)
                        "model_config": checkpoint_model_config,
                        "quantizer_metadata": quantizer_metadata,
                        "metric_metadata": metric_metadata,
                    }
                    save_tokenizer_checkpoint(ckpt, accelerator.unwrap_model(model),
                                              Path(save_path) / f"epoch_{epoch + 1}.pt",
                                              checkpoint_contract, max_checkpoints)
                except Exception as exc:
                    error = f"Checkpoint save failed: {exc}"
            try:
                _checkpoint_error(error)
                accelerator.wait_for_everyone()
            finally:
                restore_rng_state(entry_rng, accelerator.device)

    if accelerator.is_main_process:
        log_info(phase="complete", epoch=cfg.epochs, step=global_step, output_dir=save_path)


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)

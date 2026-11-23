"""Versioned LR recipes and explicit checkpoint resume policy for the tokenizer.

The recipe, rather than LambdaLR.state_dict alone, owns the lambda's constants.
Only call scheduler.step() after a successful optimizer update. A reset restarts
the LR clock while retaining the checkpoint's AdamW state (including momentum,
betas and weight decay); it is not an optimizer-state reset or weights-only load.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch


RECIPE_VERSION = 1
FORMULA = "linear_step_plus_one_then_cosine_v1"


def resolve_scheduler_resume_mode(optim_cfg):
    explicit = optim_cfg.get("scheduler_resume_mode", None)
    reset = bool(optim_cfg.get("reset_scheduler", False))
    mode = str(explicit) if explicit is not None else ("reset" if reset else "strict_continue")
    if mode not in ("strict_continue", "reset"):
        raise ValueError("optim.scheduler_resume_mode must be 'strict_continue' or 'reset'")
    if explicit is not None and mode == "strict_continue" and reset:
        raise ValueError("scheduler_resume_mode=strict_continue conflicts with reset_scheduler=true")
    return mode


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
    return value


def _lr_vector(values, size, name, *, positive=False):
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"{name} must contain one LR per optimizer parameter group ({size})")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 or (positive and value == 0) for value in result):
        raise ValueError(f"{name} must contain finite {'positive' if positive else 'nonnegative'} LRs")
    return result


def _assert_lrs(actual, expected, label):
    actual = _lr_vector(actual, len(expected), label)
    if any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(actual, expected)):
        raise ValueError(f"{label} disagree with the saved LR recipe: actual={actual}, expected={expected}")


def make_lr_recipe(optimizer, optim_cfg, num_training_steps, start_global_step=0):
    """Build a fresh plan; optional base_lrs/min_lrs support explicit group LRs."""
    name = str(optim_cfg.get("scheduler", "none")).lower()
    if name in ("", "none", "constant"):
        name = "constant"
    if name != "constant" and name != "cosine":
        raise ValueError(f"unknown optim.scheduler={name!r} (expected cosine|none)")
    size = len(optimizer.param_groups)
    base_lrs = _lr_vector(
        list(optim_cfg["base_lrs"]) if optim_cfg.get("base_lrs") is not None else [float(optim_cfg["lr"])] * size,
        size, "base_lrs", positive=name == "cosine",
    )
    min_lrs = _lr_vector(
        list(optim_cfg["min_lrs"]) if optim_cfg.get("min_lrs") is not None else [float(optim_cfg.get("min_lr", 1e-7))] * size,
        size, "min_lrs",
    ) if name == "cosine" else list(base_lrs)
    if any(floor > base for floor, base in zip(min_lrs, base_lrs)):
        raise ValueError("min_lrs must not exceed base_lrs")
    return {
        "version": RECIPE_VERSION,
        "formula": FORMULA,
        "scheduler": name,
        "base_lrs": base_lrs,
        "min_lrs": min_lrs,
        "warmup_steps": _integer(int(optim_cfg.get("warmup_steps", 200)), "warmup_steps") if name == "cosine" else 0,
        "num_training_steps": _integer(int(num_training_steps), "num_training_steps", 1),
        "start_global_step": _integer(start_global_step, "start_global_step"),
        "completed_updates": 0,
        "optimizer_resume_policy": "fresh_optimizer",
    }


def _validate_recipe(recipe, size):
    if not isinstance(recipe, dict) or recipe.get("version") != RECIPE_VERSION or recipe.get("formula") != FORMULA:
        raise ValueError("Checkpoint lacks a supported complete LR recipe (lr_schedule). "
                         "Cannot strict_continue a reference/unknown schedule; explicitly choose "
                         "optim.scheduler_resume_mode=reset to start a new LR plan with saved optimizer state.")
    required = ("scheduler", "base_lrs", "min_lrs", "warmup_steps", "num_training_steps",
                "start_global_step", "completed_updates")
    missing = [key for key in required if key not in recipe]
    if missing:
        raise ValueError(f"Incomplete saved LR recipe: missing {missing}; strict_continue cannot guess")
    result = deepcopy(recipe)
    name = result["scheduler"]
    if name not in ("cosine", "constant"):
        raise ValueError(f"Unsupported saved scheduler {name!r}")
    result["base_lrs"] = _lr_vector(result["base_lrs"], size, "saved base_lrs", positive=name == "cosine")
    result["min_lrs"] = _lr_vector(result["min_lrs"], size, "saved min_lrs")
    if any(floor > base for floor, base in zip(result["min_lrs"], result["base_lrs"])):
        raise ValueError("Saved min_lrs exceed base_lrs")
    for key in ("warmup_steps", "start_global_step", "completed_updates"):
        _integer(result[key], f"saved {key}")
    _integer(result["num_training_steps"], "saved num_training_steps", 1)
    return result


def lr_multiplier(step, *, warmup_steps, num_training_steps, min_ratio):
    # Preserve the existing EAT warmup convention exactly: step 0 uses 1/warmup;
    # steps warmup-1 and warmup both use base LR. At step num_training_steps the
    # cosine reaches its floor (provided the requested horizon exceeds warmup).
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = min(max((step - warmup_steps) / max(num_training_steps - warmup_steps, 1), 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def recipe_lrs(recipe, completed_updates):
    if recipe["scheduler"] == "constant":
        return list(recipe["base_lrs"])
    return [base * lr_multiplier(completed_updates, warmup_steps=recipe["warmup_steps"],
                               num_training_steps=recipe["num_training_steps"], min_ratio=floor / base)
            for base, floor in zip(recipe["base_lrs"], recipe["min_lrs"])]


def _construct_scheduler(optimizer, recipe):
    # Override BOTH keys. LambdaLR otherwise reuses a checkpoint's initial_lr.
    for group, base in zip(optimizer.param_groups, recipe["base_lrs"]):
        group["lr"] = base
        group["initial_lr"] = base
    if recipe["scheduler"] == "constant":
        return None
    lambdas = [lambda step, ratio=floor / base: lr_multiplier(
        step, warmup_steps=recipe["warmup_steps"], num_training_steps=recipe["num_training_steps"],
        min_ratio=ratio) for base, floor in zip(recipe["base_lrs"], recipe["min_lrs"])]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def initialize_lr_schedule(optimizer, optim_cfg, num_training_steps, *, checkpoint=None, global_step=0):
    """Restore optimizer and schedule together, returning (scheduler, recipe).

    strict_continue ignores current YAML schedule settings and uses only the
    saved recipe. reset uses current settings and starts at local step zero;
    global_step remains the count of successful updates over the entire run.
    Checkpoints without optimizer state are rejected in either resume mode.
    """
    _integer(global_step, "global_step")
    mode = resolve_scheduler_resume_mode(optim_cfg)
    if checkpoint is None:
        if global_step != 0:
            raise ValueError("Fresh LR plan requires global_step=0")
        recipe = make_lr_recipe(optimizer, optim_cfg, num_training_steps)
        return _construct_scheduler(optimizer, recipe), recipe

    if "global_step" not in checkpoint or checkpoint["global_step"] is None:
        raise ValueError("Checkpoint lacks global_step; cannot infer successful updates from epoch/dataloader length")
    saved_step = _integer(checkpoint["global_step"], "checkpoint global_step")
    if saved_step != global_step:
        raise ValueError("global_step disagrees with the checkpoint")
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict) or not isinstance(optimizer_state.get("param_groups"), list):
        raise ValueError("Resume requires optimizer_state_dict, not missing/None; both modes preserve optimizer state")
    saved_groups = optimizer_state["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("Checkpoint optimizer parameter-group count differs from current optimizer")

    if mode == "reset":
        # Validate the new recipe before touching optimizer state.
        recipe = make_lr_recipe(optimizer, optim_cfg, num_training_steps, start_global_step=global_step)
        optimizer.load_state_dict(optimizer_state)
        recipe["optimizer_resume_policy"] = "preserve_checkpoint_optimizer_reset_lr_clock"
        return _construct_scheduler(optimizer, recipe), recipe

    recipe = _validate_recipe(checkpoint.get("lr_schedule"), len(saved_groups))
    completed = recipe["completed_updates"]
    if recipe["start_global_step"] + completed != global_step:
        raise ValueError("Saved LR start_global_step + completed_updates must equal checkpoint global_step")
    expected = recipe_lrs(recipe, completed)
    _assert_lrs([group.get("initial_lr", float("nan")) for group in saved_groups], recipe["base_lrs"],
                "checkpoint optimizer initial_lr")
    saved_lrs = [group.get("lr", float("nan")) for group in saved_groups]
    _assert_lrs(saved_lrs, expected, "checkpoint optimizer lr")
    if "scheduler_state_dict" not in checkpoint:
        raise ValueError("strict_continue requires scheduler_state_dict (explicit None for constant schedules)")
    scheduler_state = checkpoint["scheduler_state_dict"]
    if recipe["scheduler"] == "constant":
        if scheduler_state is not None:
            raise ValueError("Constant saved recipe requires scheduler_state_dict=None")
    else:
        if not isinstance(scheduler_state, dict):
            raise ValueError("Cosine strict_continue requires a non-None scheduler_state_dict")
        if scheduler_state.get("last_epoch") != completed:
            raise ValueError("Scheduler last_epoch disagrees with saved completed_updates")
        _assert_lrs(scheduler_state.get("base_lrs", []), recipe["base_lrs"], "scheduler base_lrs")
        _assert_lrs(scheduler_state.get("_last_lr", []), expected, "scheduler _last_lr")
        if scheduler_state.get("lr_lambdas") != [None] * len(saved_groups):
            raise ValueError("Unsupported saved LambdaLR callables; expected the versioned EAT function recipe")

    optimizer.load_state_dict(optimizer_state)
    scheduler = _construct_scheduler(optimizer, recipe)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    # LambdaLR construction touched optimizer.lr at step 0. Loading its state
    # does NOT restore optimizer.lr; restore the saved next-update LR explicitly.
    for group, lr in zip(optimizer.param_groups, saved_lrs):
        group["lr"] = float(lr)
    recipe["optimizer_resume_policy"] = "preserve_checkpoint_optimizer_and_lr_clock"
    return scheduler, recipe


def lr_schedule_state(optimizer, scheduler, recipe, global_step):
    """Serialize recipe + clock, checking against the LR actually held by AdamW."""
    result = _validate_recipe(recipe, len(optimizer.param_groups))
    completed = _integer(global_step, "global_step") - result["start_global_step"]
    _integer(completed, "completed_updates")
    result["completed_updates"] = completed
    if result["scheduler"] == "constant":
        if scheduler is not None:
            raise ValueError("Constant recipe unexpectedly has a scheduler")
    else:
        if scheduler is None or scheduler.last_epoch != completed:
            raise ValueError("Scheduler and successful optimizer update counts disagree at save")
        _assert_lrs(scheduler.base_lrs, result["base_lrs"], "live scheduler base_lrs")
        _assert_lrs(scheduler.get_last_lr(), recipe_lrs(result, completed), "live scheduler last_lr")
    _assert_lrs([group.get("initial_lr", float("nan")) for group in optimizer.param_groups],
                result["base_lrs"], "live optimizer initial_lr")
    _assert_lrs([group["lr"] for group in optimizer.param_groups], recipe_lrs(result, completed), "live optimizer lr")
    return result

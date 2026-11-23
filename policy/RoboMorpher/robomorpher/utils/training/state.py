"""Per-rank training RNG and AMP state, using weights-only-safe checkpoint values."""
from __future__ import annotations
import random
import numpy as np
import torch


def capture_rng_state(device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": {"algorithm": numpy_state[0], "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng_state(state, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    values = state["numpy"]
    np.random.set_state((values["algorithm"], values["keys"].cpu().numpy().astype(np.uint32),
                         values["position"], values["has_gauss"], values["cached_gaussian"]))
    if state["cuda"] is not None:
        if device.type != "cuda":
            raise ValueError("Saved CUDA RNG requires a CUDA training device for exact resume")
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def collect_tokenizer_training_state(accelerator):
    """All ranks enter; only rank zero receives the gathered checkpoint payload."""
    scaler = getattr(accelerator, "scaler", None)
    local = {"rng": capture_rng_state(accelerator.device),
             "scaler": None if scaler is None else scaler.state_dict(),
             "accumulation_step": int(getattr(accelerator, "step", 0))}
    world = int(accelerator.num_processes)
    if world > 1:
        states = [None] * world if accelerator.is_main_process else None
        torch.distributed.gather_object(local, states, dst=0)
    else:
        states = [local]
    payload = {"version": 2, "world_size": world,
               "gradient_accumulation_steps": int(accelerator.gradient_accumulation_steps),
               "mixed_precision": str(accelerator.mixed_precision), "by_rank": states}
    return payload, local["rng"]


def restore_tokenizer_training_state(checkpoint, accelerator):
    """checkpoints remain loadable but cannot promise RNG-exact continuation."""
    payload = checkpoint.get("training_state")
    if payload is None:
        return False
    if payload.get("version") != 2:
        raise ValueError("Unsupported tokenizer training_state version; exact resume requires version 2 with accumulation metadata")
    accumulation = payload.get("gradient_accumulation_steps")
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError("Tokenizer training_state requires valid gradient_accumulation_steps metadata")
    if accumulation != int(accelerator.gradient_accumulation_steps):
        raise ValueError("Exact tokenizer resume requires the same gradient_accumulation_steps")
    if payload.get("world_size") != int(accelerator.num_processes):
        raise ValueError("Exact tokenizer RNG/AMP resume requires the same world size")
    if payload.get("mixed_precision") != str(accelerator.mixed_precision):
        raise ValueError("Exact tokenizer RNG/AMP resume requires the same mixed_precision")
    states = payload.get("by_rank")
    if not isinstance(states, list) or len(states) != int(accelerator.num_processes):
        raise ValueError("Invalid tokenizer per-rank training state")
    state = states[int(accelerator.process_index)]
    saved_step = state.get("accumulation_step")
    if isinstance(saved_step, bool) or not isinstance(saved_step, int) or saved_step < 0 or saved_step % accumulation:
        raise ValueError("Tokenizer checkpoint accumulation_step is not a completed accumulation boundary")
    if (state["rng"].get("cuda") is not None) != (torch.device(accelerator.device).type == "cuda"):
        raise ValueError("Exact tokenizer RNG resume requires the same CPU/CUDA device type")
    scaler = getattr(accelerator, "scaler", None)
    if (state.get("scaler") is None) != (scaler is None):
        raise ValueError("Checkpoint GradScaler presence differs from this training run")
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    accelerator.step = saved_step
    restore_rng_state(state["rng"], accelerator.device)
    return True

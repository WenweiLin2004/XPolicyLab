"""Flatten hierarchical ``vqvae_model`` yaml into the tokenizer constructor kwargs."""

from __future__ import annotations

import warnings

_MISSING = object()

# One latent per joint, quantized with residual VQ.
STEP_ARCHITECTURES = {
    "joint_rvq": ("joint_features", "rvq"),
}


def _as_dict(cfg) -> dict:
    if isinstance(cfg, dict):
        return {k: _as_dict(v) for k, v in cfg.items()}
    if hasattr(cfg, "keys"):
        return {k: _as_dict(cfg[k]) for k in cfg.keys()}
    return cfg


def _nested_get(cfg: dict, path, default=None):
    cur = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def canonical_architecture_name(value, cfg=None) -> str:
    """Return the joint residual-VQ name stored by current checkpoints."""
    del cfg
    name = str(value).lower()
    if name != "joint_rvq":
        raise ValueError("tokenizer architecture must be joint_rvq")
    return name


def architecture_signature(value, cfg=None) -> str:
    """Return the canonical semantic assembly for resume compatibility."""
    return canonical_architecture_name(value, cfg)


def _get(c, *paths, default=None, cast=None):
    """First existing path wins. ``paths`` are nested tuples or flat key strings."""
    val = _MISSING
    for path in paths:
        if isinstance(path, str):
            path = (path,)
        found = _nested_get(c, path, _MISSING)
        if found is not _MISSING:
            val = found
            break
    if val is _MISSING:
        val = default
    if cast is not None and val is not None:
        return cast(val)
    return val


def normalize_block_style(value) -> str:
    """Canonical block layout. The previous checkpoint label maps to the MLP residual."""
    name = str(value).lower()
    if name not in {"mlp_attention_residual", "standard_prenorm"}:
        raise ValueError("frame block style must be mlp_attention_residual or standard_prenorm")
    return name


def normalize_query_style(value) -> str:
    """Query style name used by the frame codec."""
    return str(value).lower()


def normalize_model_config(m) -> dict:
    """Hierarchical ``vqvae_model.<group>.*`` plus flat keys."""
    c = _as_dict(m)
    gb = _get(c, ("encoder", "graph_bias"), default={})
    gb = gb if isinstance(gb, dict) else {}
    ent = _get(c, ("quantizer", "entropy"), default={})
    ent = ent if isinstance(ent, dict) else {}
    recon = _get(c, ("losses", "recon"), default={})
    recon = recon if isinstance(recon, dict) else {}
    ee = _get(c, ("losses", "ee"), default={})
    ee = ee if isinstance(ee, dict) else {}
    joint = _get(c, ("losses", "joint"), default={})
    joint = joint if isinstance(joint, dict) else {}

    raw_architecture = str(_get(c, "architecture", default="joint_rvq")).lower()
    out = {
        "architecture": canonical_architecture_name(raw_architecture, c),
        "joint_info_dim": _get(c, ("input", "joint_info_dim"), "joint_info_dim", default=10, cast=int),
        "urdf_rotation_6d": _get(c, ("input", "urdf_rotation_6d"), "urdf_rotation_6d", default=False, cast=bool),
        "n_head": _get(c, ("encoder", "n_head"), "n_head", default=8, cast=int),
        "quant_proj_layers": _get(c, ("encoder", "quant_proj_layers"), "quant_proj_layers", default=4, cast=int),
        "embed_mlp_hidden": _get(c, ("encoder", "embed_mlp_hidden"), "embed_mlp_hidden", default=1024, cast=int),
        "urdf_layers": _get(c, ("encoder", "urdf_layers"), "urdf_layers", default=0, cast=int),
        "state_layers": _get(c, ("encoder", "state_layers"), "state_layers", default=0, cast=int),
        "angle_layers": _get(c, ("encoder", "angle_layers"), "angle_layers", default=0, cast=int),
        "angle_mlp_hidden": _get(c, ("encoder", "angle_mlp_hidden"), "angle_mlp_hidden", default=256, cast=int),
        "urdf_angle_fusion": str(
            _get(c, ("encoder", "fusion"), "urdf_angle_fusion", default="add")
        ).lower(),
        "use_spatial_encoding": bool(gb.get("spatial", _get(c, "use_spatial_encoding", default=True))),
        "use_parent_encoding": bool(gb.get("parent", _get(c, "use_parent_encoding", default=False))),
        "use_child_encoding": bool(gb.get("child", _get(c, "use_child_encoding", default=False))),
        "graph_bias_zero_init": bool(gb.get("zero_init", _get(c, "graph_bias_zero_init", default=False))),
        "use_zero_init_gate": _get(
            c, ("encoder", "zero_init_gate", "enable"), "use_zero_init_gate", default=False, cast=bool
        ),
        "codebook_dim": _get(c, ("quantizer", "codebook_dim"), "codebook_dim", default=512, cast=int),
        "codebook_size": _get(c, ("quantizer", "codebook_size"), "codebook_size", default=512, cast=int),
        "num_codebooks": _get(c, ("quantizer", "num_codebooks"), "num_codebooks", default=2, cast=int),
        "beta": _get(c, ("quantizer", "beta"), "beta", default=0.25, cast=float),
        "use_ema_quantizer": _get(c, ("quantizer", "ema", "enable"), "use_ema_quantizer", default=False, cast=bool),
        "quantizer_backend": str(
            _get(c, ("quantizer", "backend"), "quantizer_backend", default="auto")
        ).lower(),
        "ema_decay": _get(c, ("quantizer", "ema", "decay"), "ema_decay", default=0.8, cast=float),
        # Standalone/evaluation construction retains the historical default.
        # The training entry point resolves fresh-run and resume policy below.
        "ema_update_unit": str(
            _get(c, ("quantizer", "ema", "update_unit"), "ema_update_unit", default="microbatch_forward")
        ).lower(),
        "kmeans_iters": _get(c, ("quantizer", "ema", "kmeans_iters"), "kmeans_iters", default=100, cast=int),
        "dead_code_threshold": _get(
            c, ("quantizer", "ema", "dead_code_threshold"), "dead_code_threshold", default=2.0, cast=float
        ),
        "use_rotation_trick": _get(
            c, ("quantizer", "rotation_trick", "enable"), "use_rotation_trick", default=True, cast=bool
        ),
        "decoder_ee_dim": _get(c, ("decoder", "ee_dim"), "decoder_ee_dim", default=9, cast=int),
        "decoder_joint_dim": _get(c, ("decoder", "joint_dim"), "decoder_joint_dim", default=1, cast=int),
        "post_quant_proj_layers": _get(
            c, ("decoder", "post_quant_proj_layers"), "post_quant_proj_layers", default=4, cast=int
        ),
        "decoder_mlp_hidden": _get(c, ("decoder", "mlp_hidden"), "decoder_mlp_hidden", default=1024, cast=int),
        "dual_branch_decoder": _get(
            c, ("decoder", "dual_branch", "enable"), "dual_branch_decoder", default=False, cast=bool
        ),
        "max_num_joints": _get(c, "max_num_joints", default=8, cast=int),
        "load_dir": _get(c, "load_dir"),
    }

    recon_w = float(recon.get("weight", _get(c, "recon_loss_weight", default=1.0)))
    out["recon_loss_weight"] = recon_w
    # losses.ee / losses.joint; fall back to recon.weight so old 1:1 yamls stay unchanged
    out["ee_loss_weight"] = float(ee.get("weight", recon.get("ee_weight", recon_w)))
    out["joint_loss_weight"] = float(joint.get("weight", recon.get("joint_weight", recon_w)))

    ent_weight = float(ent.get("weight", _get(c, "entropy_loss_weight", default=0.0)))
    out["entropy_loss_weight"] = ent_weight
    out["use_entropy_loss"] = bool(ent.get("enable", ent_weight > 0))
    out["entropy_temp"] = float(ent.get("temp", _get(c, "entropy_temp", default=0.01)))

    out["decoder_out_dim"] = int(
        _get(c, ("decoder", "out_dim"), default=out["decoder_ee_dim"] + out["decoder_joint_dim"])
    )
    out["decoder_ee_layers"] = int(
        _get(c, ("decoder", "dual_branch", "ee_layers"), default=out["post_quant_proj_layers"])
    )
    out["decoder_joint_layers"] = int(
        _get(c, ("decoder", "dual_branch", "joint_layers"), default=out["post_quant_proj_layers"])
    )

    out["n_latent_dims"] = int(
        _get(c, ("encoder", "n_latent_dims"), "n_latent_dims", default=out["codebook_dim"])
    )
    if out["architecture"] in STEP_ARCHITECTURES:
        for field, paths, default in (
            ("output_with_ee", (("decoder", "output_with_ee"), "output_with_ee"), True),
            ("urdf_injection_enabled", (("encoder", "urdf_injection", "enable"), "urdf_injection_enabled"), True),
        ):
            value = _get(c, *paths, default=default)
            if not isinstance(value, bool):
                raise ValueError(f"{field} must be a YAML boolean")
            out[field] = value
        architecture_defaults = STEP_ARCHITECTURES.get(out["architecture"])
        requested_quantizer = _get(c, ("quantizer", "type"), "quantizer_type", default=None)
        out["quantizer_type"] = str(
            requested_quantizer if requested_quantizer is not None
            else (architecture_defaults[1] if architecture_defaults else "rvq")
        ).lower()
        if out["quantizer_type"] != "rvq":
            raise ValueError("quantizer.type must be rvq")
        if architecture_defaults and out["quantizer_type"] != architecture_defaults[1]:
            raise ValueError(
                f"{out['architecture']} requires quantizer.type={architecture_defaults[1]!r}"
            )
        # EE-only codecs may omit the joint-angle head (output is exactly EE9).
        requested_joint_dim = int(_get(c, ("decoder", "joint_dim"), "decoder_joint_dim", default=1))
        if requested_joint_dim < 0:
            raise ValueError("decoder.joint_dim must be nonnegative")
        expected_output_dim = 9 + requested_joint_dim if out["output_with_ee"] else requested_joint_dim
        requested_output_dim = _get(c, ("decoder", "out_dim"), "decoder_out_dim")
        if requested_output_dim is not None and int(requested_output_dim) != expected_output_dim:
            raise ValueError("decoder.out_dim disagrees with decoder.output_with_ee/joint_dim")
        out["decoder_joint_dim"] = requested_joint_dim
        out["decoder_out_dim"] = expected_output_dim
        state_input_mode = str(_get(
            c, ("encoder", "state_input_mode"), "state_input_mode", default="angle"
        )).lower()
        if state_input_mode != "angle":
            raise ValueError("encoder.state_input_mode must be angle")
        expected_input_dim = 7
        requested_input_dim = int(_get(c, ("input", "joint_info_dim"), "joint_info_dim", default=expected_input_dim))
        if requested_input_dim != expected_input_dim:
            raise ValueError(
                f"state_input_mode={state_input_mode} requires input.joint_info_dim={expected_input_dim}, got {requested_input_dim}")
        out["joint_info_dim"] = requested_input_dim
        input_with_ee = _get(c, ("encoder", "input_with_ee"), "input_with_ee", default=False)
        if input_with_ee is not False:
            raise ValueError("encoder.input_with_ee is not used; the encoder takes joint angles")
        latent_default = architecture_defaults[0] if architecture_defaults else "learn_query"
        requested_latent = _get(c, ("frame_codec", "latent_mode"), "frame_latent_mode", default=None)
        latent_mode = str(requested_latent if requested_latent is not None else latent_default).lower()
        if latent_mode != "joint_features":
            raise ValueError("frame_codec.latent_mode must be joint_features")
        if architecture_defaults and latent_mode != architecture_defaults[0]:
            raise ValueError(
                f"{out['architecture']} requires frame_codec.latent_mode={architecture_defaults[0]!r}"
            )
        requested_latents = _get(c, ("frame_codec", "num_latents"), "frame_num_latents", cast=int)
        if latent_mode == "joint_features":
            # Node latents retain the padded joint layout; an old query count
            # must never determine the new quantizer shape or token budget.
            frame_num_latents = out["max_num_joints"]
            if requested_latents is not None and requested_latents != frame_num_latents:
                warnings.warn(
                    "joint_features derives num_latents from max_num_joints="
                    f"{frame_num_latents}; ignoring configured num_latents={requested_latents}",
                    UserWarning, stacklevel=2,
                )
        else:
            frame_num_latents = 4 if requested_latents is None else requested_latents
        out.update({
            "frame_latent_mode": latent_mode,
            "input_with_ee": input_with_ee,
            "state_input_mode": state_input_mode,
            "frame_encoder_attention": str(_get(
                c, ("encoder", "attention"), "frame_encoder_attention", default="causal"
            )).lower(),
            "frame_num_latents": frame_num_latents,
            "rvq_num_quantizers": _get(c, ("quantizer", "num_quantizers"), "rvq_num_quantizers", default=4, cast=int),
            "query_num_heads": _get(c, ("frame_codec", "num_heads"), "query_num_heads", default=out["n_head"], cast=int),
            "frame_query_ffn_multiplier": _get(c, ("frame_codec", "ffn_multiplier"), "frame_query_ffn_multiplier", default=4, cast=int),
            "frame_block_style": normalize_block_style(_get(
                c, ("frame_codec", "block_style"), "frame_block_style", default="mlp_attention_residual"
            )),
            "frame_decoder_attention": str(_get(
                c, ("frame_codec", "decoder_attention"), "frame_decoder_attention", default="causal"
            )).lower(),
            "frame_query_style": normalize_query_style(_get(
                c, ("frame_codec", "query_style"), "frame_query_style", default="small_query"
            )),
            "dead_code_fraction": _get(c, ("quantizer", "ema", "dead_code_fraction"), "dead_code_fraction", default=0.1, cast=float),
        })
    return out


def resolve_training_model_config(m, checkpoint=None) -> dict:
    """Prefer upstream accumulation-boundary EMA for new training runs.

    An explicit YAML/CLI value may switch the policy deliberately. Old VQP
    checkpoints without this field used per-forward EMA. Never infer a new
    default for a resume just because the current software default changed.
    The removed custom optimizer_step mode requires an explicit policy change.
    """
    c = _as_dict(m)
    requested_arch = normalize_model_config(c)["architecture"]
    if checkpoint is not None:
        saved_config = checkpoint.get("model_config") or {}
        saved_arch = normalize_model_config(saved_config)["architecture"]
        metadata_arch = (checkpoint.get("quantizer_metadata") or {}).get("architecture")
        saved_signature = architecture_signature(saved_arch, saved_config)
        requested_signature = architecture_signature(requested_arch, c)
        if metadata_arch is not None:
            metadata_signature = architecture_signature(metadata_arch, saved_config)
            if metadata_signature != saved_signature:
                raise ValueError("Checkpoint architecture disagrees between metadata and model_config")
        if requested_signature != saved_signature:
            raise ValueError(
                f"Cannot resume {saved_arch!r} checkpoint as {requested_arch!r}; "
                "start a fresh run or perform an explicit migration"
            )
        if requested_arch in STEP_ARCHITECTURES:
            requested_model = normalize_model_config(c)
            saved_model = normalize_model_config(saved_config)
            metadata_mode = (checkpoint.get("quantizer_metadata") or {}).get("latent_mode")
            if metadata_mode is not None and metadata_mode != saved_model["frame_latent_mode"]:
                raise ValueError("Checkpoint latent mode disagrees between metadata and model_config")
            for field in ("frame_latent_mode", "input_with_ee", "output_with_ee", "urdf_injection_enabled", "quantizer_type", "frame_encoder_attention", "frame_decoder_attention", "frame_block_style", "urdf_angle_fusion", "use_spatial_encoding", "use_parent_encoding", "use_child_encoding"):
                if requested_model[field] != saved_model[field]:
                    raise ValueError(
                        f"Cannot resume with changed {field}: checkpoint={saved_model[field]!r}, "
                        f"requested={requested_model[field]!r}; start a fresh run"
                    )
    # Historical configs/checkpoints may still describe FK supervision. It is
    # no longer a training objective; do not persist these stale switches in
    # a newly saved model_config. Architecture/state_dict keys are unchanged.
    for key in ("use_fk", "fk_loss_weight", "fk_warmup_steps", "fk_stop_grad_quant", "fk_huber_beta"):
        c.pop(key, None)
    if isinstance(c.get("losses"), dict):
        c["losses"].pop("fk", None)
    if not normalize_model_config(c)["use_ema_quantizer"]:
        return c
    requested = _get(c, ("quantizer", "ema", "update_unit"), "ema_update_unit")
    persisted = None
    if checkpoint is not None:
        metadata = checkpoint.get("quantizer_metadata") or {}
        saved = _as_dict(checkpoint.get("model_config") or {})
        saved_unit = _get(saved, ("quantizer", "ema", "update_unit"), "ema_update_unit")
        persisted = metadata.get("ema_update_unit") or saved_unit or "microbatch_forward"
        if saved_unit is not None and str(saved_unit).lower() != str(persisted).lower():
            raise ValueError("Checkpoint EMA policy disagrees between metadata and model_config")
    unit = str(requested if requested is not None else persisted or "accumulation_boundary").lower()
    if unit == "optimizer_step":
        raise ValueError(
            "The custom optimizer_step EMA mode has been removed. Explicitly set "
            "quantizer.ema.update_unit=accumulation_boundary to use the upstream "
            "forward implementation, or microbatch_forward for per-forward EMA. "
            "Switching a checkpoint's policy is not an identical training resume."
        )
    if unit not in {"microbatch_forward", "accumulation_boundary"}:
        raise ValueError(f"Unknown ema_update_unit={unit!r}; expected microbatch_forward|accumulation_boundary")
    c["ema_update_unit"] = unit
    if isinstance(c.get("quantizer"), dict):
        ema = c["quantizer"].setdefault("ema", {})
        if not isinstance(ema, dict):
            raise ValueError("quantizer.ema must be a mapping")
        ema["update_unit"] = unit
    return c

"""Construct assembled tokenizer architectures from normalized config."""
from __future__ import annotations

from .config import STEP_ARCHITECTURES, normalize_model_config
from ..quantizers.residual import _resolve_quantizer_backend
from ..robomorpher import RoboMorpher

_BUILD_KEYS = (
    "joint_info_dim", "n_latent_dims", "codebook_dim", "codebook_size",
    "num_codebooks", "n_head", "decoder_out_dim", "quant_proj_layers",
    "post_quant_proj_layers", "embed_mlp_hidden", "decoder_mlp_hidden",
    "use_spatial_encoding", "use_parent_encoding", "use_child_encoding",
    "graph_bias_zero_init", "use_zero_init_gate", "entropy_temp",
    "use_entropy_loss", "use_ema_quantizer", "quantizer_backend",
    "ema_decay", "ema_update_unit", "kmeans_iters", "dead_code_threshold",
    "use_rotation_trick", "dual_branch_decoder", "decoder_ee_layers",
    "decoder_joint_layers", "beta", "urdf_layers", "state_layers",
    "angle_layers", "angle_mlp_hidden", "urdf_angle_fusion",
)


def _step_constructor(architecture, _normalized):
    if architecture != "joint_rvq":
        raise ValueError(f"Unknown tokenizer architecture {architecture!r}")
    return RoboMorpher


def build_tokenizer_from_config(m, state_dict=None):
    """Build a complete tokenizer assembly without implicit PQ/RVQ migration."""
    m = normalize_model_config(m)
    m["quantizer_backend"] = _resolve_quantizer_backend(
        m["quantizer_backend"], m["use_ema_quantizer"], state_dict
    )
    if m.get("state_input_mode") not in (None, "angle"):
        raise ValueError("encoder.state_input_mode must be angle")
    expected = 10 if m["urdf_rotation_6d"] else 7
    if int(m["joint_info_dim"]) != expected:
        raise ValueError(
            f"joint_info_dim={int(m['joint_info_dim'])} but urdf_rotation_6d="
            f"{m['urdf_rotation_6d']} requires {expected}"
        )
    kwargs = {k: m[k] for k in _BUILD_KEYS}
    kwargs["num_joints"] = m["max_num_joints"]
    kwargs["ee_dim"] = m["decoder_ee_dim"]
    kwargs["joint_dim"] = m["decoder_joint_dim"]
    architecture = m.get("architecture", "joint_rvq")
    step_architecture = architecture in STEP_ARCHITECTURES
    is_step_state = state_dict is not None and "_frame_schema" in state_dict
    if state_dict is not None and is_step_state != step_architecture:
        raise ValueError(
            "Checkpoint architecture differs from requested tokenizer; "
            "no implicit PQ/RVQ or frame/joint migration"
        )
    if step_architecture:
        for name in (
            "frame_num_latents", "rvq_num_quantizers", "query_num_heads",
            "frame_query_ffn_multiplier", "frame_block_style",
            "frame_decoder_attention", "frame_query_style", "dead_code_fraction",
            "input_with_ee", "state_input_mode", "frame_encoder_attention", "frame_latent_mode",
            "output_with_ee", "urdf_injection_enabled", "quantizer_type",
        ):
            kwargs[name] = m[name]
        return _step_constructor(architecture, m)(**kwargs)
    raise ValueError(f"Unknown tokenizer architecture {architecture!r}")

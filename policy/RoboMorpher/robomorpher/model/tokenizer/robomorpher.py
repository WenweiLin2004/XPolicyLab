"""Single-frame graph tokenizer with per-joint latents and shared-stage RVQ."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch import distributed as dist
from robomorpher.utils.model.masking import binary_bool_mask
from .joint_tokenizer import JointTokenizer
from .components.heads import DecodeHead
from .components.embeddings import JointEmbed, AngleEmbed
from .components.attention import LlamaRMSNorm, process_stack
from .components.fusion import NormalizedAddFusion
from .utils.config import normalize_block_style, normalize_query_style
from .quantizers.residual import SharedFrameRVQ, _resolve_quantizer_backend


class RoboMorpher(JointTokenizer):
    architecture = "joint_rvq"

    def __init__(
        self, joint_info_dim=7, n_latent_dims=2048, codebook_dim=2048,
        codebook_size=512, num_codebooks=4, n_head=16, num_joints=7,
        decoder_out_dim=10, ee_dim=9, joint_dim=1, quant_proj_layers=8,
        post_quant_proj_layers=4, embed_mlp_hidden=1024, decoder_mlp_hidden=1024,
        use_spatial_encoding=True, use_parent_encoding=False, use_child_encoding=False,
        graph_bias_zero_init=False, use_zero_init_gate=False, entropy_temp=0.01,
        use_entropy_loss=False, use_ema_quantizer=True, quantizer_backend="vqp",
        ema_decay=0.8, ema_update_unit="accumulation_boundary", kmeans_iters=100,
        dead_code_threshold=8.0, use_rotation_trick=True, dual_branch_decoder=False,
        decoder_ee_layers=None, decoder_joint_layers=None, beta=0.25,
        urdf_layers=2, state_layers=4, angle_layers=2, angle_mlp_hidden=256,
        urdf_angle_fusion="normalized_add", frame_num_latents=4,
        rvq_num_quantizers=4, query_num_heads=16, frame_query_ffn_multiplier=4,
        frame_block_style="mlp_attention_residual", frame_decoder_attention="causal",
        frame_query_style="small_query", frame_latent_mode="joint_features",
        input_with_ee=False, frame_encoder_attention="causal",
        state_input_mode="angle",
        dead_code_fraction=0.1, output_with_ee=True, urdf_injection_enabled=True,
        quantizer_type="rvq", **kwargs,
    ):
        nn.Module.__init__(self)
        self.frame_latent_mode = str(frame_latent_mode).lower()
        if self.frame_latent_mode != "joint_features":
            raise ValueError("frame_latent_mode must be joint_features")
        if self.frame_latent_mode == "joint_features":
            # This mode has one latent for each padded joint, not learned slots.
            frame_num_latents = num_joints
        if not isinstance(input_with_ee, bool):
            raise ValueError("input_with_ee must be a boolean")
        self.input_with_ee = input_with_ee
        self.state_input_mode = str(state_input_mode).lower()
        if self.input_with_ee or self.state_input_mode != "angle":
            raise ValueError("the encoder takes joint angles, not end-effector pose")
        if not isinstance(output_with_ee, bool) or not isinstance(urdf_injection_enabled, bool):
            raise ValueError("output_with_ee and urdf_injection_enabled must be booleans")
        self.output_with_ee = output_with_ee
        self.urdf_injection_enabled = urdf_injection_enabled
        # These dimensions are used by the output schema validation below.
        # Initialize them before the EE-only decoder fallback accesses joint_dim.
        self.ee_dim = int(ee_dim)
        self.joint_dim = int(joint_dim)
        self.quantizer_type = str(quantizer_type).lower()
        if self.quantizer_type != "rvq":
            raise ValueError("quantizer_type must be rvq")
        # EE-only codecs use a 9-D decoder (xyz+rot6d) with no joint-angle head.
        decoder_out_dim = self.joint_dim if not self.output_with_ee else decoder_out_dim
        self.frame_encoder_attention = str(frame_encoder_attention).lower()
        if self.frame_encoder_attention not in {"causal", "bidirectional"}:
            raise ValueError("encoder attention must be causal or bidirectional")
        if ee_dim != 9 or joint_dim < 0:
            raise ValueError("Frame codec requires EE9 and a nonnegative joint output dimension")
        expected_input_dim = (6 + ee_dim) if (self.state_input_mode == "ee" and self.urdf_injection_enabled) else (ee_dim if self.state_input_mode == "ee" else 7)
        expected_output_dim = (ee_dim + joint_dim) if self.output_with_ee else joint_dim
        if int(joint_info_dim) != expected_input_dim or int(decoder_out_dim) != expected_output_dim:
            raise ValueError(
                f"Frame codec input/output schema mismatch: expected input {expected_input_dim}, "
                f"output {expected_output_dim}, got input {joint_info_dim}, output {decoder_out_dim}")
        if min(n_latent_dims, codebook_dim, n_head) < 1:
            raise ValueError("Hidden/codebook widths and attention head counts must be positive")
        if n_latent_dims % n_head:
            raise ValueError("Hidden width must be divisible by the graph attention head count")
        if min(num_joints, frame_num_latents, rvq_num_quantizers) < 1:
            raise ValueError("Joint/latent/stage counts must be positive")
        if num_codebooks != rvq_num_quantizers:
            raise ValueError("num_codebooks must equal rvq_num_quantizers (one table per stage)")
        if not use_ema_quantizer or dual_branch_decoder or use_entropy_loss:
            raise ValueError("Frame RVQ requires EMA, a single decoder, and no entropy objective")
        if (self.urdf_injection_enabled and urdf_layers < 1) or angle_layers < 1 or state_layers < 1 or post_quant_proj_layers < 1:
            raise ValueError("Both input streams, fusion stack and joint decoder must be enabled")
        if urdf_angle_fusion not in {"normalized_add", "adaln_zero"}:
            raise ValueError("Frame RVQ fusion must be normalized_add or adaln_zero")
        self.urdf_conditioning = self.urdf_injection_enabled and urdf_angle_fusion == "adaln_zero"
        self.frame_block_style = normalize_block_style(frame_block_style)
        self.frame_decoder_attention = str(frame_decoder_attention).lower()
        if self.frame_decoder_attention not in {"causal", "bidirectional"}:
            raise ValueError("decoder attention must be causal or bidirectional")
        self.standard_residual = self.frame_block_style == "standard_prenorm"
        if self.urdf_conditioning and (not self.standard_residual or use_zero_init_gate):
            raise ValueError(
                "adaln_zero requires standard_prenorm and zero_init_gate.enable=false; "
                "its own conditional residual gates must not multiply old zero gates"
            )
        self.frame_query_style = normalize_query_style(frame_query_style)
        if self.frame_query_style not in {"small_query", "learned_query_rope"}:
            raise ValueError("frame query style must be small_query or learned_query_rope")
        self.quantizer_backend = _resolve_quantizer_backend(quantizer_backend, True)
        self.codebook_dim = int(codebook_dim)
        self.n_latent_dims = int(n_latent_dims)
        self.codebook_size = int(codebook_size)  # PER STAGE, never divided by R.
        self.num_codebooks = int(rvq_num_quantizers)
        self.rvq_num_quantizers = int(rvq_num_quantizers)
        self.frame_num_latents = int(frame_num_latents)
        self.tokens_per_frame = self.frame_num_latents * self.rvq_num_quantizers
        self.n_head = int(n_head)
        self.num_joints = int(num_joints)
        self.ee_dim, self.joint_dim, self.decoder_out_dim = ee_dim, joint_dim, decoder_out_dim
        self.dual_branch_decoder = False
        self.use_ema_quantizer = True
        self.joint_info_dim = int(joint_info_dim)
        self.urdf_dim = 6
        self.state_input_dim = (self.ee_dim if self.state_input_mode == "ee"
                                else self.ee_dim + self.joint_dim if self.input_with_ee
                                else self.joint_dim)
        self.urdf_layers, self.angle_layers, self.state_layers = urdf_layers, angle_layers, state_layers
        self.urdf_angle_fusion = urdf_angle_fusion
        self.use_urdf_split = True
        self.dead_code_fraction = float(dead_code_fraction)
        self._quantizer_settings = dict(
            backend="vqp", architecture=self.architecture, layout="residual_full_vector",
            num_codebooks=self.num_codebooks, num_latents=self.frame_num_latents,
            codebook_dim=self.codebook_dim, codebook_size=self.codebook_size,
            distance="raw_l2", gradient_estimator="rotation_trick" if use_rotation_trick else "ste",
            ema_update_unit=ema_update_unit, ema_decay=float(ema_decay),
            dead_code_threshold=float(dead_code_threshold), commitment_weight=float(beta),
        )
        self.bias_cfg = None
        if use_spatial_encoding or use_parent_encoding or use_child_encoding:
            self.bias_cfg = dict(num_joints=num_joints, num_heads=n_head,
                                 enable_spatial=use_spatial_encoding, enable_parent=use_parent_encoding,
                                 enable_child=use_child_encoding, zero_init=graph_bias_zero_init)
        self._build_encoder(embed_mlp_hidden, quant_proj_layers, angle_mlp_hidden, use_zero_init_gate)
        self.fusion = (NormalizedAddFusion(self.n_latent_dims)
                       if self.urdf_injection_enabled and not self.urdf_conditioning else None)
        # One shared pair around the ENTIRE residual stack, as in ResidualVQ.
        # Identity modules preserve the existing H == D checkpoint layout.
        self.has_projections = self.n_latent_dims != self.codebook_dim
        self.project_in = (nn.Linear(self.n_latent_dims, self.codebook_dim)
                           if self.has_projections else nn.Identity())
        self.project_out = (nn.Linear(self.codebook_dim, self.n_latent_dims)
                            if self.has_projections else nn.Identity())
        self.quantizer = SharedFrameRVQ(
            dim=self.codebook_dim, num_latents=self.frame_num_latents,
            num_quantizers=self.rvq_num_quantizers, codebook_size=self.codebook_size,
            beta=beta, ema_decay=ema_decay, kmeans_iters=kmeans_iters,
            dead_code_threshold=dead_code_threshold, use_rotation_trick=use_rotation_trick,
            ema_update_unit=ema_update_unit,
        )
        self._build_decoder(post_quant_proj_layers, decoder_mlp_hidden,
                            decoder_ee_layers, decoder_joint_layers, use_zero_init_gate)
        schema = [
            1, self.num_joints, self.rvq_num_quantizers, self.codebook_dim, self.codebook_size,
            self.n_latent_dims, int(self.standard_residual),
            int(self.frame_encoder_attention == "bidirectional"),
            int(self.frame_decoder_attention == "bidirectional"),
            int(self.urdf_conditioning), int(self.output_with_ee), int(self.urdf_injection_enabled),
            self.joint_dim,
        ]
        self.register_buffer("_frame_schema", torch.tensor(schema, dtype=torch.int64))

    def _width_stack(self, n, use_zero_init_gate):
        """Frame codec blocks use a standard pre-norm identity residual."""
        return process_stack(
            n, self.n_latent_dims, self.n_latent_dims, self.n_head,
            use_zero_init_gate, bias_cfg=self.bias_cfg,
            standard_residual=self.standard_residual,
        )

    def _build_encoder(self, embed_hidden, quant_proj_layers, angle_mlp_hidden, use_zero_init_gate):
        if not self.urdf_injection_enabled:
            self.angle_embed = AngleEmbed(self.n_latent_dims, angle_mlp_hidden, self.state_input_dim)
            self.angle_proj = self._width_stack(self.angle_layers, use_zero_init_gate)
            self.state_proj = self._width_stack(self.state_layers, use_zero_init_gate)
            self.urdf_angle_film = None
            return
        if not self.urdf_conditioning:
            return super()._build_encoder(embed_hidden, quant_proj_layers, angle_mlp_hidden, use_zero_init_gate)
        width = self.n_latent_dims
        self.urdf_embed = JointEmbed(self.urdf_dim, width, embed_hidden)
        self.angle_embed = AngleEmbed(width, angle_mlp_hidden, self.state_input_dim)
        # Encode each input stream once; only the state refinement stack is conditional.
        self.urdf_proj = self._width_stack(self.urdf_layers, False)
        self.angle_proj = self._width_stack(self.angle_layers, False)
        self.state_proj = process_stack(
            self.state_layers, width, width, self.n_head, bias_cfg=self.bias_cfg,
            standard_residual=True, condition_dim=width,
        )
        self.urdf_angle_film = None

    def _build_decoder(self, post_layers, mlp_hidden, ee_layers, joint_layers, use_zero_init_gate):
        # project_out restores the hidden width before this stack.
        self.post_quant_proj = process_stack(
            post_layers, self.n_latent_dims, self.n_latent_dims, self.n_head,
            use_zero_init_gate, min_layers=1, bias_cfg=self.bias_cfg,
            standard_residual=self.standard_residual,
            condition_dim=self.n_latent_dims if self.urdf_conditioning else None,
        )
        self.decoder_final_norm = (
            LlamaRMSNorm(self.n_latent_dims) if self.standard_residual else nn.Identity()
        )
        self.decoder_head = DecodeHead(self.n_latent_dims, mlp_hidden, self.decoder_out_dim)

    def quantizer_metadata(self):
        metadata = super().quantizer_metadata()
        metadata.update(
            architecture=self.architecture,
            schema_version=1,
            hidden_dim=self.n_latent_dims, codebook_dim=self.codebook_dim,
            projection_layout="shared_pre_post_rvq" if self.has_projections else "identity",
            residual_space_dim=self.codebook_dim,
            id_schema="frame_slot_major_stage_fast_local_ids_v1", padding_id=None,
            tokens_per_frame=self.tokens_per_frame,
            bits_per_frame=self.tokens_per_frame * math.log2(self.codebook_size),
            commitment_reduction="global_valid_latent_mean_then_stage_mean",
            ema_sample_unit="valid_frames_times_latent_slots_per_stage",
            fusion="independent_rmsnorm_add", continuous_input_bypass=False,
            query_style=self.frame_query_style,
            input_with_ee=self.input_with_ee,
            encoder_attention=self.frame_encoder_attention,
            encoder_state_features=("normalized_ee_xyz_rot6d" if self.state_input_mode == "ee" else "normalized_ee_xyz_rot6d_then_angle" if self.input_with_ee else "angle"),
            state_input_mode=self.state_input_mode,
            decoder_requires_ee=False,
            encoder_block=("standard_prenorm_identity_residual" if self.standard_residual
                           else "mlp_attention_residual"),
            decoder_structure=(
                "joint_query_cross_attention_then_bidirectional_joint_self_attention"
                if self.frame_decoder_attention == "bidirectional" else
                "joint_query_cross_attention_then_causal_joint_self_attention"
            ),
        )
        metadata["latent_mode"] = self.frame_latent_mode
        if self.frame_latent_mode == "joint_features":
            metadata.update(
                id_schema="joint_slot_major_stage_fast_local_ids_v1", padding_id=-1,
                tokens_per_frame_is_padded_maximum=True,
                valid_tokens_per_frame="num_valid_joints_times_num_stages",
                bits_per_frame_is_padded_maximum=True,
                ema_sample_unit="valid_joint_slots_per_stage",
                query_style=None,
                decoder_structure=(
                    "joint_aligned_bidirectional_joint_self_attention"
                    if self.frame_decoder_attention == "bidirectional" else
                    "joint_aligned_causal_joint_self_attention"
                ),
            )
        if self.state_input_mode != "angle" or self.joint_dim != 1:
            metadata.update(state_input_mode=self.state_input_mode, joint_info_dim=self.joint_info_dim, joint_dim=self.joint_dim, output_dim=self.decoder_out_dim)
        if self.urdf_conditioning:
            metadata.update(
                fusion="per_joint_adaln_zero",
                conditioning="normalized_urdf_xyz_rpy_encoded_once_per_forward",
                conditioning_dim=self.n_latent_dims,
                conditioning_layers=dict(encoder=len(self.state_proj), decoder=len(self.post_quant_proj)),
                conditioning_parameter_sharing="shared_urdf_encoder_independent_per_block_modulation",
                decoder_requires_urdf=True, decoder_requires_ee=False,
                continuous_input_bypass=True, continuous_decoder_features="urdf_only_no_ee_or_angle",
                encoder_block="adaln_zero_identity_residual",
            )
        metadata.update(quantizer_type=self.quantizer_type, output_with_ee=self.output_with_ee,
                        urdf_injection_enabled=self.urdf_injection_enabled)
        if not self.urdf_injection_enabled:
            metadata.update(fusion="none", decoder_requires_urdf=False, continuous_input_bypass=False,
                            conditioning="disabled", encoder_block=("standard_prenorm_identity_residual"
                            if self.standard_residual else "mlp_attention_residual"))
        return metadata

    def _legacy_joint_schema(self):
        """Schema markers used by historical joint-aligned RVQ checkpoints."""
        if self.state_input_mode != "angle" or self.joint_dim != 1:
            return None
        layout = [self.num_joints, self.frame_num_latents, self.rvq_num_quantizers,
                  self.codebook_dim, self.codebook_size, self.n_latent_dims, int(self.input_with_ee)]
        encoder_bidir = int(self.frame_encoder_attention == "bidirectional")
        decoder_bidir = int(self.frame_decoder_attention == "bidirectional")
        if not self.output_with_ee or not self.urdf_injection_enabled:
            return [10, *layout, int(self.output_with_ee), int(self.urdf_injection_enabled),
                    int(self.urdf_conditioning), 0, 1, int(self.standard_residual), encoder_bidir,
                    decoder_bidir, 0]
        if self.urdf_conditioning:
            return [9, *layout, encoder_bidir, decoder_bidir, 1, 0]
        return None

    def _check_schema(self, state, prefix=""):
        key = prefix + "_frame_schema"
        if key not in state:
            raise ValueError("This is not a frame-RVQ checkpoint; PQ checkpoints cannot be migrated implicitly")
        incoming = state[key]
        expected = self._frame_schema.detach().cpu()
        if (torch.is_tensor(incoming) and incoming.dtype == torch.int64 and incoming.ndim == 1
                and incoming.numel() != expected.numel()
                and incoming.tolist() == self._legacy_joint_schema()):
            state[key] = self._frame_schema.detach().clone()
            return
        if not torch.is_tensor(incoming) or tuple(incoming.shape) != tuple(expected.shape):
            raise ValueError("Frame-RVQ checkpoint projection/schema mismatch")
        if not torch.equal(incoming.detach().cpu(), expected):
            raise ValueError("Frame-RVQ checkpoint latent/stage/width/vocabulary schema mismatch")

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        self._require_completed_ema_window()
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._require_completed_ema_window()
        self._check_schema(state_dict, prefix)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                             missing_keys, unexpected_keys, error_msgs)

    def train(self, mode=True):
        if not mode:
            self._require_completed_ema_window()
        return super().train(mode)

    def _joint_mask(self, batch, device, mask):
        if mask is None:
            return torch.ones(batch, self.num_joints, dtype=torch.bool, device=device)
        mask = binary_bool_mask(mask, device=device, name="joint_valid_mask")
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(batch, -1)
        if tuple(mask.shape) != (batch, self.num_joints):
            raise ValueError(f"joint_valid_mask must have shape [B,{self.num_joints}]")
        return mask

    def _prepare_inputs(self, features, joint_valid_mask):
        if features.ndim != 3 or tuple(features.shape[1:]) != (self.num_joints, self.joint_info_dim):
            raise ValueError(f"Frame encoder requires [B,{self.num_joints},{self.joint_info_dim}] padded inputs")
        mask = self._joint_mask(features.shape[0], features.device, joint_valid_mask)
        clean = features.masked_fill(~mask[..., None], 0.0)
        if not self.urdf_injection_enabled and self.state_input_mode == "angle":
            clean = torch.cat((torch.zeros_like(clean[..., :6]), clean[..., 6:]), dim=-1)
        invalid = (~mask.any(-1)).any() | (~torch.isfinite(clean)).any()
        # Fail on every training rank before any codebook collective begins.
        flag = invalid.to(dtype=torch.int64)
        if self.training and dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if bool(flag):
            raise ValueError("Frame batch contains an all-padding frame or non-finite valid features")
        return clean, mask

    def _prepare_ee(self, ee_features, features, mask):
        del features, mask
        if ee_features is not None:
            raise ValueError("the encoder takes joint angles and does not accept ee_features")
        return None

    def _apply_attn_layers(self, x, layers, joint_valid_mask):
        if self.frame_encoder_attention == "causal":
            return super()._apply_attn_layers(x, layers, joint_valid_mask)
        valid = self._joint_mask(x.shape[0], x.device, joint_valid_mask)
        blocked_keys = (~valid)[:, None, :].expand(-1, x.shape[1], -1)
        x = x.masked_fill(~valid[..., None], 0.0)
        for layer in layers:
            x = layer(x, attn_mask=blocked_keys)
            x = x.masked_fill(~valid[..., None], 0.0)
        return x

    def encode_urdf(self, urdf, joint_valid_mask=None):
        if not self.urdf_injection_enabled:
            raise RuntimeError("URDF encoder is disabled by encoder.urdf_injection.enable=false")
        return super().encode_urdf(urdf, joint_valid_mask)

    def encode_angle(self, angle, joint_valid_mask=None):
        return super().encode_angle(angle, joint_valid_mask)

    def _apply_conditioned_layers(self, x, layers, condition, mask, *, causal, modulations=None):
        blocked = (~mask)[:, None, :].expand(-1, self.num_joints, -1)
        if causal:
            positions = torch.arange(self.num_joints, device=x.device)
            blocked = blocked | (positions[None, :] > positions[:, None])[None]
        x = x.masked_fill(~mask[..., None], 0.0)
        if modulations is not None and len(modulations) != len(layers):
            raise ValueError("Decoder requires one modulation per layer")
        for i, layer in enumerate(layers):
            kwargs = {} if modulations is None else {"modulation": modulations[i]}
            x = layer(x, attn_mask=blocked, condition=condition, joint_valid_mask=mask, **kwargs)
            x = x.masked_fill(~mask[..., None], 0.0)
        return x

    def _encode_hidden_and_condition(self, features, joint_valid_mask=None, *, ee_features=None):
        """Keep the condition local to this call; never cache a previous batch's URDF."""
        clean, mask = self._prepare_inputs(features, joint_valid_mask)
        clean_ee = self._prepare_ee(ee_features, clean, mask)
        geometry = self.encode_urdf(clean[..., :6], mask) if self.urdf_injection_enabled else None
        del clean_ee
        state = clean[..., self.urdf_dim:] if self.urdf_injection_enabled else clean[..., -self.state_input_dim:]
        angles = self.encode_angle(state, mask)
        if self.urdf_conditioning:
            nodes = self._apply_conditioned_layers(
                angles, self.state_proj, geometry, mask,
                causal=self.frame_encoder_attention == "causal",
            )
        else:
            nodes = self.fusion(geometry, angles, mask) if self.urdf_injection_enabled else angles
            nodes = self._apply_attn_layers(nodes, self.state_proj, mask)
        nodes = nodes.masked_fill(~mask[..., None], 0.0)
        return nodes, geometry if self.urdf_conditioning else None

    def encode_hidden(self, features, joint_valid_mask=None, *, ee_features=None):
        """Continuous [B,S,H] features; S is padded J in joint_features mode."""
        return self._encode_hidden_and_condition(features, joint_valid_mask, ee_features=ee_features)[0]

    def _latent_mask(self, joint_mask):
        """One latent per padded joint, so validity is the joint mask."""
        return joint_mask

    def _encode_with_condition(self, features, joint_valid_mask=None, *, ee_features=None):
        hidden, condition = self._encode_hidden_and_condition(
            features, joint_valid_mask, ee_features=ee_features,
        )
        z = self.project_in(hidden)
        if self.frame_latent_mode == "joint_features":
            mask = self._joint_mask(features.shape[0], features.device, joint_valid_mask)
            z = z.masked_fill(~mask[..., None], 0.0)
        return z, condition

    def _encode(self, features, joint_valid_mask=None, *, ee_features=None):
        return self._encode_with_condition(features, joint_valid_mask, ee_features=ee_features)[0]

    def encode_latent(self, features, joint_valid_mask=None, *, ee_features=None):
        """Continuous VQ input [B,S,D]; use this space for quantization bypass."""
        return self._encode(features, joint_valid_mask, ee_features=ee_features)

    def _decoder_condition(self, z, mask, urdf_features, urdf_condition):
        if not self.urdf_conditioning:
            if urdf_features is not None or urdf_condition is not None:
                raise ValueError("normalized_add decoder does not accept URDF conditioning")
            return None
        if (urdf_features is None) == (urdf_condition is None):
            raise ValueError("adaln_zero decoder requires exactly one of urdf_features or urdf_condition")
        raw = urdf_condition if urdf_condition is not None else urdf_features
        width = self.n_latent_dims if urdf_condition is not None else self.urdf_dim
        expected = (z.shape[0], self.num_joints, width)
        if (not torch.is_tensor(raw) or tuple(raw.shape) != expected
                or raw.device != z.device or not torch.is_floating_point(raw)):
            raise ValueError(f"URDF decoder condition must be a floating tensor {expected} on the latent device")
        clean = raw.masked_fill(~mask[..., None], 0.0)
        if not bool(mask.any(-1).all()) or not bool(torch.isfinite(clean).all()):
            raise ValueError("URDF decoder condition contains an all-padding frame or non-finite valid values")
        if urdf_condition is None:
            # Public discrete decoding only accepts normalized xyz+rpy, never target EE/q.
            clean = self.encode_urdf(clean.to(dtype=next(self.urdf_embed.parameters()).dtype), mask)
        return clean

    def _decode(self, z, joint_valid_mask=None, *, urdf_features=None, urdf_condition=None, return_features=False, decoder_modulations=None):
        if z.ndim != 3 or tuple(z.shape[1:]) != (self.frame_num_latents, self.codebook_dim):
            raise ValueError("Decoder requires [B,num_latents,codebook_dim], before shared project_out")
        mask = self._joint_mask(z.shape[0], z.device, joint_valid_mask)
        condition = self._decoder_condition(z, mask, urdf_features, urdf_condition)
        if self.frame_latent_mode == "joint_features":
            z = z.masked_fill(~mask[..., None], 0.0)
        z = self._decoder_input(z)
        # Apply AFTER the sum of all RVQ stages, exactly once (including bias).
        nodes = self.project_out(z).masked_fill(~mask[..., None], 0.0)
        if self.urdf_conditioning:
            nodes = self._apply_conditioned_layers(
                nodes, self.post_quant_proj, condition, mask,
                causal=self.frame_decoder_attention == "causal", modulations=decoder_modulations,
            )
        elif self.frame_decoder_attention == "causal":
            # Decoder attention must not inherit the encoder's new switch.
            nodes = super()._apply_attn_layers(nodes, self.post_quant_proj, mask)
        else:
            # Reconstruction sees the whole frame and refines all output
            # queries with bidirectional attention.
            blocked_keys = (~mask)[:, None, :].expand(-1, self.num_joints, -1)
            for layer in self.post_quant_proj:
                nodes = layer(nodes, attn_mask=blocked_keys)
                nodes = nodes.masked_fill(~mask[..., None], 0.0)
        nodes = self.decoder_final_norm(nodes).masked_fill(~mask[..., None], 0.0)
        if return_features:
            return nodes
        return self.decoder_head(nodes).masked_fill(~mask[..., None], 0.0)

    @property
    def decoder_feature_dim(self):
        """Final decoder-attention width before the reconstruction MLP."""
        return int(self.n_latent_dims)

    def _split_output(self, out):
        if not self.output_with_ee:
            return out[..., :0], out[..., :self.joint_dim]
        return super()._split_output(out)

    def _apply_valid_mask(self, ee_out, joint_out, joint_valid_mask):
        mask = self._joint_mask(ee_out.shape[0], ee_out.device, joint_valid_mask)
        return ee_out.masked_fill(~mask[..., None], 0.0), joint_out.masked_fill(~mask[..., None], 0.0)

    def forward(self, features, joint_valid_mask=None, *, accum_ema_update=None,
                bypass_quantizer=False, ee_features=None):
        mask = self._joint_mask(features.shape[0], features.device, joint_valid_mask)
        z, condition = self._encode_with_condition(features, mask, ee_features=ee_features)
        if bypass_quantizer:
            if accum_ema_update not in (None, False):
                raise ValueError("AE warm-up cannot accumulate an EMA update")
            zero = z.sum() * 0.0
            ee, angle = self._split_output(self._decode(z, mask, urdf_condition=condition))
            return ee, angle, zero, zero, zero.detach()
        latent_valid = self._latent_mask(mask)
        z_q, loss, entropy, usage = self.quantizer(z, latent_valid, accum_ema_update=accum_ema_update)
        ee, angle = self._split_output(self._decode(z_q, mask, urdf_condition=condition))
        return ee, angle, loss, entropy, usage

    def decode_joint(self, z, joint_valid_mask=None, *, urdf_features=None):
        return self._split_output(self._decode(z, joint_valid_mask, urdf_features=urdf_features))[1]

    def _require_frozen_codec(self):
        self._require_completed_ema_window()
        if self.training:
            raise RuntimeError("Discrete encode/decode interfaces require eval(); use forward() for training")

    @torch.no_grad()
    def action_to_idx(self, features, joint_valid_mask=None, *, ee_features=None):
        self._require_frozen_codec()
        mask = self._joint_mask(features.shape[0], features.device, joint_valid_mask)
        z = self._encode(features, mask, ee_features=ee_features)
        ids, codes = self.quantizer.f_to_idx(z, mask=self._latent_mask(mask))
        return ids.reshape(features.shape[0], self.tokens_per_frame), codes

    @torch.no_grad()
    def encode_ids(self, features, joint_valid_mask=None, *, ee_features=None):
        return self.action_to_idx(features, joint_valid_mask, ee_features=ee_features)[0]

    @torch.no_grad()
    def decode_ids(self, indices, joint_valid_mask=None, *, urdf_features=None):
        """Decode IDs; adaln_zero additionally requires normalized [B,J,6] URDF."""
        self._require_frozen_codec()
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Frame IDs must be integer tensors")
        if indices.ndim == 2 and indices.shape[1] == self.tokens_per_frame:
            indices = indices.reshape(indices.shape[0], self.frame_num_latents, self.rvq_num_quantizers)
        elif indices.ndim != 3 or tuple(indices.shape[1:]) != (self.frame_num_latents, self.rvq_num_quantizers):
            raise ValueError("Frame IDs must be [B,tokens_per_frame] or [B,num_latents,num_stages]")
        if self.frame_latent_mode == "joint_features":
            padding = indices == -1
            if bool((padding.any(-1) != padding.all(-1)).any()):
                raise ValueError("A padded joint must use -1 in every residual stage")
            inferred_mask = ~padding.all(-1)
            mask = (inferred_mask if joint_valid_mask is None else
                    self._joint_mask(indices.shape[0], indices.device, joint_valid_mask))
            if not torch.equal(mask, inferred_mask):
                raise ValueError("joint_valid_mask must match joints with -1 in every residual stage")
            if not bool(mask.any(-1).all()):
                raise ValueError("Cannot decode an all-padding frame")
            z = self.quantizer.idx_to_f(indices, mask=mask)
        else:
            # Every valid frame has all S query slots: -1 is never a query token.
            if bool(((indices < 0) | (indices >= self.codebook_size)).any()):
                raise ValueError("Frame IDs must be in [0, codebook_size); padding IDs are not allowed")
            mask = self._joint_mask(indices.shape[0], indices.device, joint_valid_mask)
            if not bool(mask.any(-1).all()):
                raise ValueError("Cannot decode an all-padding frame")
            z = self.quantizer.idx_to_f(indices)
        return self._split_output(self._decode(z, mask, urdf_features=urdf_features))

    def _decode_indices(self, indices, joint_valid_mask=None, *, urdf_features=None):
        return self.decode_ids(indices, joint_valid_mask, urdf_features=urdf_features)

    @torch.no_grad()
    def idx_to_action(self, indices, joint_valid_mask=None, *, urdf_features=None):
        return self.decode_ids(indices, joint_valid_mask, urdf_features=urdf_features)[1][..., 0]

    @torch.no_grad()
    def idx_to_ee(self, indices, joint_valid_mask=None, *, urdf_features=None):
        return self.decode_ids(indices, joint_valid_mask, urdf_features=urdf_features)[0]


__all__ = ["RoboMorpher"]

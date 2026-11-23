"""Discrete-Forcing style Hybrid DiT.

One module owns a shared trunk and two independent branches. The two action
heads can call forward_branch on the same module with branch-specific
sequence inputs, timestep encodings and output projections.
"""
from __future__ import annotations
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .categorical_dit import DiT


class HybridBranchView(nn.Module):
    """Non-registered view exposing one branch of a shared HybridActionDiT.

    The action model and continuous refiner must keep one parameter owner, but
    inspection and optimizer code expects ``model.transformer_blocks``
    to show the branch used by that head. This proxy delegates all execution
    to the shared core without registering a second module tree.
    """
    def __init__(self, core, branch: str):
        nn.Module.__init__(self)
        if branch not in {"continuous", "discrete"}:
            raise ValueError("branch must be continuous or discrete")
        # Keep this view out of the module tree; the Hybrid owner is already
        # registered by QwenDiscreteDiffusion.action_model.
        object.__setattr__(self, "_core", core)
        object.__setattr__(self, "_branch", branch)

    @property
    def is_hybrid(self):
        return self._core.is_hybrid

    @property
    def transformer_blocks(self):
        branch = (self._core.continuous_transformer_blocks
                  if self._branch == "continuous"
                  else self._core.discrete_transformer_blocks)
        return list(self._core.shared_transformer_blocks) + list(branch)

    @property
    def gradient_checkpointing(self):
        return self._core.gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._core.gradient_checkpointing = bool(value)

    def forward_branch(self, *args, **kwargs):
        return self._core.forward_branch(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._core, name)


class HybridActionDiT(nn.Module):
    """Shared trunk + discrete and continuous transformer branches.

    The caller builds either a continuous or discrete sequence and this module
    runs the same shared module objects followed by only that branch. This
    mirrors upstream DF when branch isolation and disjoint routing are on.
    """
    is_hybrid = True

    def __init__(self, base_dit: DiT, shared_layers: int, *, continuous_output_dim: int | None = None, discrete_output_dim: int | None = None, urdf_feature_dim: int = 0,
                 conditioning_layout: str = "shared_sequence", future_tokens: int = 32,
                 max_seq_len: int | None = None, continuous_horizon: int = 50):
        super().__init__()
        if not isinstance(base_dit, DiT):
            raise TypeError("base_dit must be categorical_dit.DiT")
        total = int(base_dit.config.num_layers)
        shared_layers = int(shared_layers)
        if shared_layers < 1 or shared_layers >= total:
            raise ValueError("shared_layers must be in [1, num_layers-1]")
        self.hidden_dim = int(base_dit.inner_dim)
        self.num_layers = total
        self.num_shared_layers = shared_layers
        self.num_branch_layers = total - shared_layers
        self.num_heads = int(base_dit.config.num_attention_heads)
        self.interleave_self_attention = bool(getattr(base_dit, "_interleave_self_attention", False))
        self.model_config = base_dit.config
        self.active_branch = "discrete"
        self.gradient_checkpointing = False
        if conditioning_layout not in {"shared_sequence", "df_per_token"}:
            raise ValueError("conditioning_layout must be shared_sequence or df_per_token")
        self.conditioning_layout = conditioning_layout
        self.uses_df_token_routing = conditioning_layout == "df_per_token"
        self.continuous_horizon = int(continuous_horizon)
        if self.uses_df_token_routing:
            if int(future_tokens) < 0 or self.continuous_horizon < 1:
                raise ValueError("DF future-token count and continuous horizon are invalid")
            capacity = int(max_seq_len or base_dit.config.max_num_positional_embeddings)
            if capacity <= self.continuous_horizon:
                raise ValueError("DF position capacity must include continuous and discrete actions")
            # Only this core owns these parameters. Branch heads reference them
            # at execution time, never register aliases in their module trees.
            self.shared_future_tokens = nn.Parameter(torch.empty(int(future_tokens), self.hidden_dim))
            self.shared_position_embedding = nn.Embedding(capacity, self.hidden_dim)
            nn.init.normal_(self.shared_future_tokens, std=0.02)
            nn.init.normal_(self.shared_position_embedding.weight, std=0.02)
            self.continuous_branch_temb_proj = nn.Identity()
            self.discrete_branch_temb_proj = nn.Identity()
            self.continuous_branch_out_proj = nn.Identity()
            self.discrete_branch_out_proj = nn.Identity()

        cfg = dict(base_dit.config)
        cfg.pop("num_layers", None)
        cfg["num_layers"] = shared_layers
        cfg["output_dim"] = int(base_dit.inner_dim)
        if self.uses_df_token_routing:
            cfg["positional_embeddings"] = None
        self.shared_model = DiT(**cfg)
        # New DF branches must continue the global cross/self layer parity,
        # including when an odd number of shared layers is requested.
        cfg["num_layers"] = total if self.uses_df_token_routing else self.num_branch_layers
        cfg["output_dim"] = int(continuous_output_dim or base_dit.inner_dim)
        self.continuous_model = DiT(**cfg)
        cfg["output_dim"] = int(discrete_output_dim or base_dit.config.output_dim)
        self.discrete_model = DiT(**cfg)
        if self.uses_df_token_routing:
            for model in (self.continuous_model, self.discrete_model):
                model.transformer_blocks = nn.ModuleList(list(model.transformer_blocks)[shared_layers:])
        # These are non-registered aliases. Registering the same ModuleList and
        # timestep modules both under their owner and under convenience aliases
        # makes FSDP traverse the same child twice and assert when the second
        # path is wrapped. The owner paths (shared_model/continuous_model/
        # discrete_model) remain in the state dict and optimizer.
        object.__setattr__(self, "shared_transformer_blocks", self.shared_model.transformer_blocks)
        object.__setattr__(self, "continuous_transformer_blocks", self.continuous_model.transformer_blocks)
        object.__setattr__(self, "discrete_transformer_blocks", self.discrete_model.transformer_blocks)
        object.__setattr__(self, "continuous_timestep_encoder", self.continuous_model.timestep_encoder)
        object.__setattr__(self, "discrete_timestep_encoder", self.discrete_model.timestep_encoder)

    @property
    def transformer_blocks(self):
        # Compatibility view for heads. It is a true shared-trunk then
        # branch stack; code that needs distinct timestep semantics should call
        # forward_branch directly.
        branch = self.continuous_transformer_blocks if self.active_branch == "continuous" else self.discrete_transformer_blocks
        return list(self.shared_transformer_blocks) + list(branch)

    @property
    def timestep_encoder(self):
        return (self.continuous_timestep_encoder if self.active_branch == "continuous"
                else self.discrete_timestep_encoder)

    @property
    def norm_out(self):
        return (self.continuous_model.norm_out if self.active_branch == "continuous"
                else self.discrete_model.norm_out)

    @property
    def proj_out_1(self):
        return (self.continuous_model.proj_out_1 if self.active_branch == "continuous"
                else self.discrete_model.proj_out_1)

    @property
    def proj_out_2(self):
        return (self.continuous_model.proj_out_2 if self.active_branch == "continuous"
                else self.discrete_model.proj_out_2)

    @classmethod
    def from_dit(cls, base_dit: DiT, shared_layers: int, *, continuous_output_dim: int | None = None, discrete_output_dim: int | None = None, urdf_feature_dim: int = 0,
                 conditioning_layout: str = "shared_sequence", future_tokens: int = 32,
                 max_seq_len: int | None = None, continuous_horizon: int = 50):
        out = cls(base_dit, shared_layers, continuous_output_dim=continuous_output_dim, discrete_output_dim=discrete_output_dim, urdf_feature_dim=urdf_feature_dim,
                  conditioning_layout=conditioning_layout, future_tokens=future_tokens,
                  max_seq_len=max_seq_len, continuous_horizon=continuous_horizon)
        for idx in range(int(shared_layers)):
            out.shared_transformer_blocks[idx] = base_dit.transformer_blocks[idx]
        for idx in range(int(shared_layers), int(base_dit.config.num_layers)):
            out.discrete_transformer_blocks[idx - int(shared_layers)] = base_dit.transformer_blocks[idx]
        if not out.uses_df_token_routing:
            out.shared_model.timestep_encoder.load_state_dict(base_dit.timestep_encoder.state_dict())
            out.discrete_model.timestep_encoder.load_state_dict(base_dit.timestep_encoder.state_dict())
            out.discrete_model.norm_out = base_dit.norm_out
            out.discrete_model.proj_out_1 = base_dit.proj_out_1
            out.discrete_model.proj_out_2 = base_dit.proj_out_2
        for idx, block in enumerate(base_dit.transformer_blocks):
            modulation = getattr(block, 'urdf_modulation', None)
            if modulation is not None:
                target = out.shared_transformer_blocks[idx] if idx < int(shared_layers) else out.discrete_transformer_blocks[idx - int(shared_layers)]
                target.urdf_modulation = modulation
        if out.uses_df_token_routing:
            for model in (out.shared_model, out.continuous_model, out.discrete_model):
                # The official head calls blocks then a branch decoder. Its
                # generic DiT output AdaLN/projection is not part of that path.
                for name in ("norm_out", "proj_out_1", "proj_out_2"):
                    delattr(model, name)
                for block in model.transformer_blocks:
                    block.pos_embed = None
        return out

    def action_positions(self, branch, length, device):
        if not self.uses_df_token_routing:
            raise RuntimeError("Shared DF positions are only defined for df_per_token")
        if branch not in {"continuous", "discrete"}:
            raise ValueError("Invalid action branch")
        offset = self.continuous_horizon if branch == "discrete" else 0
        if int(length) < 1 or offset + int(length) > self.shared_position_embedding.num_embeddings:
            raise ValueError("Shared DF action position capacity is too small")
        return torch.arange(offset, offset + int(length), device=device)

    def branch_view(self, branch):
        return HybridBranchView(self, branch)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        for model in (self.shared_model, self.continuous_model, self.discrete_model):
            model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        for model in (self.shared_model, self.continuous_model, self.discrete_model):
            model.gradient_checkpointing = False

    @staticmethod
    def _run_block(block, hidden, *, encoder_hidden_states, encoder_attention_mask,
                   attention_mask, temb, urdf_condition=None, urdf_mask=None,
                   urdf_modulation=None, position_ids=None, checkpointing=False, training=False):
        kwargs = dict(
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=attention_mask,
            temb=temb,
        )
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if urdf_condition is not None:
            kwargs.update(urdf_condition=urdf_condition, urdf_mask=urdf_mask,
                          urdf_modulation=urdf_modulation)
        if checkpointing and training:
            return checkpoint(
                partial(block, **kwargs), hidden, use_reentrant=False
            )
        return block(hidden, **kwargs)

    def _run_stack(self, blocks, *, hidden, vl_embs_list, global_start,
                   temb, valid_mask, cross_mask, self_mask,
                   urdf_condition=None, urdf_modulations=None, frozen_prefix=None, frozen_prefix_len=0,
                   position_ids=None):
        layers = list(vl_embs_list) if isinstance(vl_embs_list, (list, tuple)) else [vl_embs_list] * self.num_layers
        if len(layers) < self.num_layers:
            raise ValueError("Hybrid DiT requires one VLM layer per global DiT layer")
        for local_idx, block in enumerate(blocks):
            global_idx = int(global_start) + local_idx
            is_self = self.interleave_self_attention and global_idx % 2 == 1
            encoder = None if is_self else layers[global_idx]
            modulation = urdf_modulations[global_idx] if urdf_modulations is not None else None
            hidden = self._run_block(
                block, hidden, encoder_hidden_states=encoder,
                # BasicTransformerBlock routes its attention mask through
                # encoder_attention_mask for both self and cross attention.
                encoder_attention_mask=self_mask if is_self else cross_mask,
                attention_mask=None,
                temb=temb, urdf_condition=urdf_condition,
                urdf_mask=valid_mask if urdf_condition is not None else None,
                urdf_modulation=modulation,
                position_ids=position_ids,
                checkpointing=(self.gradient_checkpointing or getattr(block, "gradient_checkpointing", False)),
                training=self.training,
            )
            hidden = hidden.masked_fill(~valid_mask[..., None], 0)
            if frozen_prefix is not None and int(frozen_prefix_len) > 0:
                hidden = torch.cat((frozen_prefix, hidden[:, int(frozen_prefix_len):]), dim=1)
        return hidden

    def forward_branch(
        self, branch: str, hidden, vl_embs_list=None, valid_mask=None,
        *, full_valid=None, global_timestep=None, branch_timestep=None,
        continuous_timestep=None, discrete_timestep=None,
        self_mask=None, cross_mask=None,
        self_attention_mask=None, cross_attention_mask=None,
        urdf_condition=None, urdf_mask=None, urdf_modulations=None,
        frozen_prefix=None, frozen_prefix_len=0,
        position_ids=None, return_hidden=False, prefix_len=None,
    ):
        """Run shared blocks and one branch, returning branch hidden states."""
        if full_valid is not None:
            valid_mask = full_valid
        if valid_mask is None:
            raise ValueError("Hybrid forward_branch requires full_valid/valid_mask")
        if self_attention_mask is not None:
            self_mask = self_attention_mask
        if cross_attention_mask is not None:
            cross_mask = cross_attention_mask
        if global_timestep is None:
            raise ValueError("Hybrid forward_branch requires global_timestep")
        if branch_timestep is None:
            branch_timestep = (continuous_timestep if branch == "continuous" else discrete_timestep)
        if branch_timestep is None:
            branch_timestep = global_timestep
        if branch not in {"continuous", "discrete"}:
            raise ValueError("branch must be continuous or discrete")
        if hidden.ndim != 3 or valid_mask.shape != hidden.shape[:2]:
            raise ValueError("Hybrid hidden/valid mask shape mismatch")
        shared_temb = self.shared_model.timestep_encoder(global_timestep.to(hidden.device))
        # Each branch must explicitly provide its own AdaLN modules. This
        # avoids silently falling back to shared modulation parameters.
        branch_encoder = self.continuous_timestep_encoder if branch == "continuous" else self.discrete_timestep_encoder
        branch_temb = branch_encoder(branch_timestep.to(hidden.device))
        if self.uses_df_token_routing:
            if prefix_len is None or not 0 <= int(prefix_len) <= hidden.shape[1]:
                raise ValueError("DF per-token routing requires the actual prefix_len")
            prefix_len = int(prefix_len)
            shared_temb = torch.cat((
                shared_temb[:, None].expand(-1, prefix_len, -1),
                branch_temb[:, None].expand(-1, hidden.shape[1] - prefix_len, -1),
            ), dim=1)
            projector = (self.continuous_branch_temb_proj if branch == "continuous"
                         else self.discrete_branch_temb_proj)
            branch_temb = projector(shared_temb)
        shared = self._run_stack(
            self.shared_transformer_blocks, hidden=hidden,
            vl_embs_list=vl_embs_list, global_start=0, temb=shared_temb,
            valid_mask=valid_mask, cross_mask=cross_mask, self_mask=self_mask,
            urdf_condition=urdf_condition,
            urdf_modulations=urdf_modulations,
            frozen_prefix=frozen_prefix, frozen_prefix_len=frozen_prefix_len,
            position_ids=position_ids,
        )
        blocks = self.continuous_transformer_blocks if branch == "continuous" else self.discrete_transformer_blocks
        out = self._run_stack(
            blocks, hidden=shared, vl_embs_list=vl_embs_list,
            global_start=self.num_shared_layers, temb=branch_temb,
            valid_mask=valid_mask, cross_mask=cross_mask, self_mask=self_mask,
            urdf_condition=urdf_condition,
            urdf_modulations=urdf_modulations,
            frozen_prefix=frozen_prefix, frozen_prefix_len=frozen_prefix_len,
            position_ids=position_ids,
        )
        if self.uses_df_token_routing:
            projector = (self.continuous_branch_out_proj if branch == "continuous"
                         else self.discrete_branch_out_proj)
            projected = projector(out)
            return (projected, out) if return_hidden else projected
        model = self.continuous_model if branch == "continuous" else self.discrete_model
        norm = model.norm_out(out)
        shift, scale = model.proj_out_1(F.silu(branch_temb)).chunk(2, dim=1)
        norm = norm * (1 + scale[:, None]) + shift[:, None]
        # Return the branch hidden representation. Each head owns its output
        # adapter: continuous_action_head applies its action decoder, while
        # discrete_code_head applies the categorical projection. This mirrors
        # the upstream HybridDiT contract and prevents double projection.
        if return_hidden:
            return norm, out
        return norm

"""Causal FAST sequence head using the existing action DiT / Hybrid DiT.

FAST labels describe the entire chunk, rather than joint/time slots.  The
input is BOS followed by the previous labels, and the output vocabulary is
the FAST vocabulary plus EOS.  BOS and PAD are input-only embeddings.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import math

import torch
from torch import nn
from torch.nn import functional as F

from robomorpher.utils.model.masking import binary_bool_mask


class FastAutoregressiveHead(nn.Module):
    def __init__(
        self, *, vocab_size: int, max_tokens: int, vl_hidden_dim: int,
        num_layers: int, hidden_dim: int | None = None,
        num_heads: int | None = None, state_dim: int = 0,
        dropout: float | None = None, diffusion_model_cfg: Mapping | None = None,
        add_pos_embed: bool = True, bos_id: int | None = None,
        pad_id: int | None = None, eos_id: int | None = None,
    ):
        super().__init__()
        for name, value in (("vocab_size", vocab_size), ("max_tokens", max_tokens),
                            ("vl_hidden_dim", vl_hidden_dim), ("num_layers", num_layers)):
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_layers < 2:
            raise ValueError("FAST autoregression requires cross and self attention (num_layers >= 2)")
        if isinstance(state_dim, bool) or int(state_dim) != state_dim or state_dim < 0:
            raise ValueError("state_dim must be a nonnegative integer")
        self.vocab_size, self.max_tokens = int(vocab_size), int(max_tokens)
        self.eos_id = int(vocab_size if eos_id is None else eos_id)
        self.bos_id = int(vocab_size + 1 if bos_id is None else bos_id)
        self.pad_id = int(vocab_size + 2 if pad_id is None else pad_id)
        if (self.eos_id, self.bos_id, self.pad_id) != (vocab_size, vocab_size + 1, vocab_size + 2):
            raise ValueError("FAST special IDs must be EOS=V, BOS=V+1, PAD=V+2")
        self.output_vocab_size, self.embedding_vocab_size = self.vocab_size + 1, self.vocab_size + 3
        self.hidden_dim = int(hidden_dim or vl_hidden_dim)
        cfg = dict(diffusion_model_cfg or {})
        head_dim = cfg.pop("attention_head_dim", None)
        self.num_heads = int(num_heads or (self.hidden_dim // int(head_dim) if head_dim else 8))
        if self.hidden_dim < 1 or self.num_heads < 1 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if head_dim is not None and int(head_dim) != self.hidden_dim // self.num_heads:
            raise ValueError("attention_head_dim conflicts with hidden_dim / num_heads")
        self.num_layers, self.vl_hidden_dim, self.state_dim = int(num_layers), int(vl_hidden_dim), int(state_dim)
        self.interleave_self_attention = True
        self.backbone_type = "categorical_dit"
        self.seq_len = self.max_tokens
        self.add_pos_embed = bool(add_pos_embed)
        for name, value in (("num_layers", self.num_layers), ("cross_attention_dim", self.vl_hidden_dim),
                            ("output_dim", self.output_vocab_size)):
            if int(cfg.pop(name, value)) != value:
                raise ValueError(f"diffusion_model_cfg.{name} conflicts with resolved dimensions")
        if not cfg.pop("interleave_self_attention", True):
            raise ValueError("FAST requires interleave_self_attention=true for causal token history")
        nested_dropout = cfg.pop("dropout", None)
        if dropout is not None and nested_dropout is not None and float(dropout) != float(nested_dropout):
            raise ValueError("dropout conflicts with diffusion_model_cfg.dropout")
        dropout = float(dropout if dropout is not None else nested_dropout if nested_dropout is not None else 0.)
        if not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be finite in [0,1]")
        allowed = {"final_dropout", "norm_type", "positional_embeddings", "max_num_positional_embeddings",
                   "attention_bias", "activation_fn", "upcast_attention", "norm_elementwise_affine", "norm_eps"}
        if set(cfg) - allowed:
            raise ValueError(f"Unsupported FAST DiT settings: {sorted(set(cfg) - allowed)}")
        required = self.max_tokens + int(self.state_dim > 0)
        capacity = int(cfg.pop("max_num_positional_embeddings", required))
        if capacity < required:
            raise ValueError("max_num_positional_embeddings cannot hold FAST and state tokens")
        from .categorical_dit import DiT
        self.model = DiT(
            num_layers=self.num_layers, num_attention_heads=self.num_heads,
            attention_head_dim=self.hidden_dim // self.num_heads,
            output_dim=self.output_vocab_size, cross_attention_dim=self.vl_hidden_dim,
            dropout=dropout, interleave_self_attention=True,
            max_num_positional_embeddings=capacity, **cfg,
        )
        self.token_embedding = nn.Embedding(self.embedding_vocab_size, self.hidden_dim, padding_idx=self.pad_id)
        self.position_embedding = nn.Embedding(capacity, self.hidden_dim) if self.add_pos_embed else None
        self.state_encoder = (nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.ReLU(),
                                            nn.Linear(self.hidden_dim, self.hidden_dim))
                              if self.state_dim else None)
        nn.init.normal_(self.token_embedding.weight, std=.02)
        with torch.no_grad():
            self.token_embedding.weight[self.pad_id].zero_()
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight, std=.02)

    @property
    def hybrid_dit(self):
        return self.model

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing = False

    @staticmethod
    def _key_mask(valid, dtype):
        return torch.zeros(valid.shape[0], 1, valid.shape[1], device=valid.device, dtype=dtype).masked_fill(
            ~valid[:, None], torch.finfo(dtype).min)

    @staticmethod
    def _causal_mask(full_valid, prefix_len, dtype):
        # Prefix queries can never read action tokens.  Otherwise the prefix
        # could carry a future target back to an earlier token at the next layer.
        length = full_valid.shape[1]
        q = torch.arange(length, device=full_valid.device)[:, None]
        k = torch.arange(length, device=full_valid.device)[None, :]
        allowed = (k <= q) | (k < prefix_len)
        allowed[:prefix_len, prefix_len:] = False
        allowed = allowed[None] & full_valid[:, None, :]
        return torch.zeros(allowed.shape, device=full_valid.device, dtype=dtype).masked_fill(
            ~allowed, torch.finfo(dtype).min)

    def _conditioning(self, vl_embs_list, batch, device, dtype, vl_attention_mask):
        layers = list(vl_embs_list) if isinstance(vl_embs_list, (list, tuple)) else [vl_embs_list] * self.num_layers
        if len(layers) < self.num_layers:
            raise ValueError("Not enough layer-wise VLM hidden states")
        layers = layers[-self.num_layers:]
        reference = next((h for h in layers if h is not None), None)
        if reference is None or reference.ndim != 3:
            raise ValueError("FAST requires VLM hidden states [B,S,D]")
        valid = (torch.ones(reference.shape[:2], dtype=torch.bool, device=device)
                 if vl_attention_mask is None else
                 binary_bool_mask(vl_attention_mask, device=device, name="vl_attention_mask"))
        if valid.ndim != 2 or valid.shape[0] != batch or not bool(valid.any(-1).all()):
            raise ValueError("VLM mask must contain at least one valid token per row")
        clean_layers = []
        for i, h in enumerate(layers):
            if h is None:
                if i % 2 == 0:
                    raise ValueError("Missing VLM state for a cross-attention layer")
                clean_layers.append(None)
                continue
            if h.ndim != 3 or tuple(h.shape[:2]) != tuple(valid.shape) or h.shape[-1] != self.vl_hidden_dim:
                raise ValueError("VLM hidden states / attention mask shape mismatch")
            h = h.to(device=device, dtype=dtype).masked_fill(~valid[..., None], 0)
            if not bool(torch.isfinite(h).all()):
                raise ValueError("Nonfinite valid VLM hidden states")
            clean_layers.append(h.contiguous())
        return clean_layers, self._key_mask(valid, dtype)

    def _logits(self, vl_embs_list, input_ids, valid_mask, state=None, vl_attention_mask=None):
        """Logits for already-shifted inputs; never accepts unshifted labels."""
        batch, length = input_ids.shape
        hidden = self.token_embedding(input_ids)
        device, dtype = hidden.device, hidden.dtype
        prefix_len = int(self.state_encoder is not None)
        if prefix_len:
            if state is None or tuple(state.shape) != (batch, self.state_dim):
                raise ValueError("FAST state must have [B,state_dim] shape")
            if not bool(torch.isfinite(state).all()):
                raise ValueError("Nonfinite FAST state")
            prefix = self.state_encoder(state.to(device=device, dtype=dtype)).to(dtype)[:, None]
            hidden = torch.cat((prefix, hidden), dim=1)
            full_valid = torch.cat((torch.ones(batch, 1, device=device, dtype=torch.bool), valid_mask), dim=1)
        else:
            full_valid = valid_mask
        uses_df = bool(getattr(self.model, "uses_df_token_routing", False))
        if uses_df:
            action_positions = self.model.action_positions(
                "discrete", hidden.shape[1] - prefix_len, device)
            hidden = hidden.clone()
            hidden[:, prefix_len:] = hidden[:, prefix_len:] + self.model.shared_position_embedding(
                action_positions)[None].to(dtype)
            positions = None
        else:
            positions = torch.arange(hidden.shape[1], device=device)
            if self.position_embedding is not None:
                hidden = hidden + self.position_embedding(positions)[None]
        hidden = hidden.masked_fill(~full_valid[..., None], 0)
        self_mask = self._causal_mask(full_valid, prefix_len, dtype)
        layers, cross_mask = self._conditioning(vl_embs_list, batch, device, dtype, vl_attention_mask)
        # One constant discrete time. FAST has no mask ratio to turn into d.
        zero_t = torch.zeros(batch, dtype=torch.long, device=device)
        if hasattr(self.model, "forward_branch"):
            hidden = self.model.forward_branch(
                "discrete", hidden, layers, full_valid=full_valid,
                self_attention_mask=self_mask, cross_attention_mask=cross_mask,
                global_timestep=zero_t, discrete_timestep=zero_t,
                continuous_timestep=zero_t, position_ids=positions,
                **({"prefix_len": prefix_len} if uses_df else {}),
            )
            if uses_df:
                if not hasattr(self, "token_decoder"):
                    raise RuntimeError("FAST df_per_token routing requires token_decoder")
                logits = self.token_decoder(hidden[:, prefix_len:])
            else:
                # Explicit branch selection also works when another caller last
                # used the shared trunk through its continuous branch view.
                owner = getattr(self.model, "discrete_model", self.model)
                logits = owner.proj_out_2(hidden[:, prefix_len:])
        else:
            temb = self.model.timestep_encoder(zero_t)
            for i, block in enumerate(self.model.transformer_blocks):
                is_self = i % 2 == 1
                call = partial(block, encoder_hidden_states=None if is_self else layers[i],
                               encoder_attention_mask=self_mask if is_self else cross_mask,
                               temb=temb, position_ids=positions)
                if self.training and self.model.gradient_checkpointing:
                    from torch.utils.checkpoint import checkpoint
                    hidden = checkpoint(call, hidden, use_reentrant=False)
                else:
                    hidden = call(hidden)
                hidden = hidden.masked_fill(~full_valid[..., None], 0)
            shift, scale = self.model.proj_out_1(F.silu(temb)).chunk(2, dim=1)
            hidden = self.model.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None]
            logits = self.model.proj_out_2(hidden[:, prefix_len:])
        return logits.masked_fill(~valid_mask[..., None], 0)

    def _check_targets(self, token_ids, valid_mask):
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("FAST token IDs must be integer")
        if token_ids.ndim != 2 or token_ids.shape[1] > self.max_tokens or token_ids.shape[1] < 1:
            raise ValueError("FAST token IDs must have [B,L] shape with 1 <= L <= max_tokens")
        valid = binary_bool_mask(valid_mask, device=token_ids.device, name="FAST valid_mask")
        if valid.shape != token_ids.shape or not bool(valid.any(-1).all()):
            raise ValueError("FAST targets need at least one valid token per row")
        if bool((~valid[:, :-1] & valid[:, 1:]).any()):
            raise ValueError("FAST targets must use right padding")
        if bool(((token_ids[valid] < 0) | (token_ids[valid] >= self.output_vocab_size)).any()):
            raise ValueError("FAST target ID outside output vocabulary")
        eos = valid & token_ids.eq(self.eos_id)
        last = valid.sum(-1) - 1
        if not bool(eos.sum(-1).eq(1).all()) or not bool(token_ids.gather(1, last[:, None]).eq(self.eos_id).all()):
            raise ValueError("FAST target must end with exactly one EOS")
        return valid

    def forward(self, vl_embs_list, token_ids, valid_mask, state=None, vl_attention_mask=None):
        valid = self._check_targets(token_ids, valid_mask)
        clean = token_ids.long().masked_fill(~valid, self.pad_id)
        shifted = torch.full_like(clean, self.bos_id)
        shifted[:, 1:] = clean[:, :-1]
        shifted = shifted.masked_fill(~valid, self.pad_id)
        logits = self._logits(vl_embs_list, shifted, valid, state, vl_attention_mask)
        losses = F.cross_entropy(logits.float().transpose(1, 2), clean.masked_fill(~valid, -100),
                                 reduction="none", ignore_index=-100)
        loss = (losses.sum(-1) / valid.sum(-1)).mean()
        accuracy = ((logits.argmax(-1).eq(clean) & valid).sum().float() / valid.sum()).detach()
        return {"action_loss": loss, "loss": loss, "logits": logits,
                "token_accuracy": accuracy, "loss_mask": valid}

    @torch.inference_mode()
    def generate(self, vl_embs_list, state=None, vl_attention_mask=None, *, max_tokens=None,
                 temperature=0.0, generator=None, logits_processor=None):
        """Autoregress until EOS, padding finished rows without target lengths.

        This correctness-first implementation recomputes the prefix at every
        step; it does not use a KV cache.  ``logits_processor(prefix, logits)``
        can enforce codec coefficient-count constraints before sampling.
        """
        budget = self.max_tokens if max_tokens is None else int(max_tokens)
        if not 1 <= budget <= self.max_tokens:
            raise ValueError("generation max_tokens outside configured capacity")
        if not math.isfinite(float(temperature)) or float(temperature) < 0:
            raise ValueError("temperature must be finite and nonnegative")
        layers = list(vl_embs_list) if isinstance(vl_embs_list, (list, tuple)) else [vl_embs_list]
        reference = next((h for h in layers if h is not None), None)
        if reference is None:
            raise ValueError("FAST generation requires VLM hidden states")
        batch = reference.shape[0]
        device = self.token_embedding.weight.device
        output = torch.full((batch, budget), self.pad_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for step in range(budget):
            inputs = torch.full((batch, step + 1), self.bos_id, dtype=torch.long, device=device)
            inputs[:, 1:] = output[:, :step]
            valid = inputs.ne(self.pad_id)
            logits = self._logits(vl_embs_list, inputs, valid, state, vl_attention_mask)[:, -1].float()
            if logits_processor is not None:
                logits = logits_processor(output[:, :step], logits)
                if logits.shape != (batch, self.output_vocab_size):
                    raise ValueError("FAST logits_processor returned the wrong shape")
            active = ~finished
            if not bool(torch.isfinite(logits[active]).any(-1).all()):
                raise ValueError("No legal FAST next token for an active sequence")
            if bool(torch.isnan(logits[active]).any()) or bool(torch.isposinf(logits[active]).any()):
                raise ValueError("Nonfinite FAST generation logits")
            # Finished rows may have all classes blocked by a strict processor.
            logits = logits.masked_fill(finished[:, None], 0)
            next_id = (logits.argmax(-1) if temperature == 0 else
                       torch.multinomial((logits / temperature).softmax(-1), 1, generator=generator)[:, 0])
            next_id = next_id.masked_fill(finished, self.pad_id)
            output[:, step] = next_id
            finished = finished | next_id.eq(self.eos_id)
            if bool(finished.all()):
                return output
        raise ValueError("FAST generation reached max_tokens without EOS; increase capacity or inspect the policy")

    predict = generate

"""Frame RVQ built from the pinned official VectorQuantize EMA implementation.

Each stage owns one full-width table shared across all latent slots. This module
only supplies residual composition and the existing EAT accumulation contract;
it does not reimplement nearest-neighbor selection, rotation trick, or EMA.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import distributed as dist
from torch.nn import functional as F

from .vector_quantize_pytorch import VectorQuantizerEMAM, _library_cdist


def _state_quantizer_backend(state_dict):
    """Recognize persisted vector-quantize-pytorch codebook keys."""
    keys = [key for key in state_dict if key.startswith("quantizer.codebooks.")]
    if not keys:
        raise ValueError("Checkpoint has no quantizer.codebooks state; supply a complete tokenizer checkpoint")
    if not any("._codebook." in key for key in keys):
        raise ValueError("Checkpoint codebook layout is not vector-quantize-pytorch")
    return "vqp"


def _resolve_quantizer_backend(requested, use_ema, state_dict=None):
    requested = str(requested).lower()
    if requested not in {"auto", "vqp"}:
        raise ValueError(f"Unknown quantizer_backend={requested!r}; expected auto|vqp")
    if state_dict is not None:
        _state_quantizer_backend(state_dict)
    if not use_ema:
        raise ValueError("The vqp backend requires use_ema_quantizer=True")
    return "vqp"


class SharedFrameRVQ(VectorQuantizerEMAM):
    """Quantize [B,S,H] into S*R local IDs, without any hidden projection.

    ``forward`` retains EAT's (quantized, commitment, zero_entropy, usage)
    contract. ``f_to_idx`` returns IDs [B,S,R] and codes [B,S,R,H]. IDs have
    no PQ-style offsets: their last axis selects the residual stage.

    All safety/precision/distributed guards are inherited from quant_vqp. The
    commitment is the mean of stage losses, each already weighted by beta once.
    """

    def __init__(
        self, dim=2048, num_latents=4, num_quantizers=4, codebook_size=512,
        beta=0.25, ema_decay=0.8, kmeans_iters=100, dead_code_threshold=2.0,
        use_rotation_trick=True, *, sync_codebook: Optional[bool] = None,
        ema_update_unit="accumulation_boundary", **kwargs,
    ):
        dim, num_latents = int(dim), int(num_latents)
        num_quantizers, codebook_size = int(num_quantizers), int(codebook_size)
        if min(dim, num_latents, num_quantizers, codebook_size) <= 0:
            raise ValueError("dim, num_latents, num_quantizers and codebook_size must be positive")
        # The base adapter initializes independent full-width VectorQuantize
        # instances. Only its safety helpers are reused; PQ slicing is not used.
        super().__init__(
            vocab_size=codebook_size * num_quantizers,
            vocab_width=dim * num_quantizers, num_codebooks=num_quantizers,
            beta=beta, ema_decay=ema_decay, kmeans_iters=kmeans_iters,
            dead_code_threshold=dead_code_threshold,
            use_rotation_trick=use_rotation_trick,
            sync_codebook=sync_codebook, ema_update_unit=ema_update_unit, **kwargs,
        )
        self.dim = self.vocab_width = dim
        self.num_latents = num_latents
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.register_buffer("last_stage_usage", torch.zeros(num_quantizers), persistent=False)

    def _features(self, features, joint_valid_mask=None, mask=None):
        if isinstance(features, torch.Tensor) and features.ndim == 3:
            if features.shape[1] != self.num_latents:
                raise ValueError(f"Frame RVQ expects {self.num_latents} latent slots, got {features.shape[1]}")
        return super()._features(features, joint_valid_mask, mask)

    def _lookup(self, clean, valid):
        """Rank-local, frozen residual lookup using the official FP32 distance."""
        ids, vectors = [], []
        residual = clean.detach()
        with torch.no_grad():
            for stage in self.codebooks:
                table = stage.codebook.detach()
                index = torch.full(valid.shape, -1, dtype=torch.long, device=clean.device)
                values = torch.zeros_like(residual)
                if bool(valid.any()):
                    nearest = _library_cdist(residual[valid].unsqueeze(0), table.unsqueeze(0)).argmin(-1)[0]
                    index[valid] = nearest
                    values[valid] = F.embedding(nearest, table)
                ids.append(index)
                vectors.append(values)
                residual = residual - values
        return ids, vectors

    def _usage(self, ids, valid, update):
        counts = torch.stack([torch.bincount(index[valid], minlength=self.group_size)
                              for index in ids]).float()
        if self.sync_codebook and update:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        probabilities = counts / counts.sum(-1, keepdim=True).clamp_min(1)
        stage_usage = (probabilities > 0.01 / self.group_size).float().mean(-1) * 100
        if update:
            with torch.no_grad():
                times = int(self.vocab_usage_record_times)
                if times == 0:
                    self.vocab_usage.copy_(probabilities)
                else:
                    self.vocab_usage.lerp_(probabilities, 0.1 if times < 100 else 0.01)
                self.vocab_usage_record_times.add_(1)
                self.last_stage_usage.copy_(stage_usage)
        return stage_usage.mean()

    def _forward_impl(self, features, joint_valid_mask=None, *, mask=None,
                      accum_ema_update: Optional[bool] = None):
        clean, valid = self._features(features, joint_valid_mask, mask)
        if self.training:
            local_count, global_count, world, accumulate = self._forward_status(clean, valid, accum_ema_update)
        else:
            if accum_ema_update is not None and accum_ema_update is not False:
                raise ValueError("Evaluation accepts only accum_ema_update=None/False")
            self._require_initialized()
            if not bool(torch.isfinite(clean).all()):
                raise ValueError("Non-finite valid VQ features")
            local_count = global_count = int(valid.sum())
            world = 1
        with torch.autocast(device_type=features.device.type, enabled=False):
            zero = clean.flatten()[:0].sum()
            if self.training and not zero.requires_grad:
                zero = zero.detach().requires_grad_()
            if global_count == 0:
                if self.training and not accumulate and self.has_pending_ema:
                    # Official empty boundary commits previously accumulated raw
                    # statistics. No samples are available for replacement.
                    with torch.no_grad():
                        for stage in self.codebooks:
                            stage(clean.detach(), mask=valid, accum_ema_update=False)
                indices = torch.full((*valid.shape, self.num_quantizers), -1,
                                     dtype=torch.long, device=clean.device)
                return clean * 0.0, indices, zero, zero, zero.detach()

            quantized = torch.zeros_like(clean)
            residual = clean
            losses = []
            if self.training:
                ids = []
                for stage in self.codebooks:
                    values, index, loss = stage(residual, mask=valid, accum_ema_update=accumulate)
                    # Use values returned by THIS stage forward. Looking up IDs
                    # here would read the table after that forward's EMA update.
                    quantized = quantized + values
                    residual = residual - values.detach()
                    ids.append(index)
                    losses.append(loss.reshape(()))
            else:
                ids, vectors = self._lookup(clean, valid)
                for values in vectors:
                    losses.append((residual[valid] - values[valid]).square().mean() * self.beta)
                    quantized = quantized + values
                    residual = residual - values
                # Preserve the source adapter's optional eval input-gradient
                # behavior; all codebook and usage buffers stay read-only.
                quantized = quantized + (clean - clean.detach())
            local_loss = torch.stack(losses).mean()
            loss = (self._global_loss(local_loss, local_count, global_count, world)
                    if self.training else local_loss)
            usage = self._usage(ids, valid, update=self.training)
            return quantized, torch.stack(ids, dim=-1), loss, zero, usage

    def forward(self, features, joint_valid_mask=None, *, mask=None,
                accum_ema_update: Optional[bool] = None):
        quantized, _, loss, zero, usage = self._forward_impl(
            features, joint_valid_mask, mask=mask, accum_ema_update=accum_ema_update,
        )
        return quantized, loss, zero, usage

    def forward_with_indices(self, features, joint_valid_mask=None, *, mask=None,
                             accum_ema_update: Optional[bool] = None):
        quantized, indices, loss, _, usage = self._forward_impl(
            features, joint_valid_mask, mask=mask, accum_ema_update=accum_ema_update,
        )
        return quantized, indices, loss, usage

    @torch.no_grad()
    def f_to_idx(self, features, joint_valid_mask=None, *, mask=None):
        clean, valid = self._features(features, joint_valid_mask, mask)
        self._require_initialized()
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("Non-finite valid VQ features")
        with torch.autocast(device_type=features.device.type, enabled=False):
            ids, vectors = self._lookup(clean, valid)
            return torch.stack(ids, dim=-1), torch.stack(vectors, dim=2)

    def idx_to_f(self, indices, joint_valid_mask=None, *, mask=None):
        self._integer_ids(indices)
        if indices.ndim != 3 or tuple(indices.shape[1:]) != (self.num_latents, self.num_quantizers):
            raise ValueError(f"RVQ indices must have shape [B,{self.num_latents},{self.num_quantizers}]")
        self._require_initialized()
        shape = indices.shape[:2]
        if joint_valid_mask is None and mask is None:
            padding = indices == -1
            if bool((padding.any(-1) != padding.all(-1)).any()):
                raise ValueError("A padded latent must use -1 in every residual stage")
            valid = ~padding.all(-1)
        else:
            valid = self._mask(shape, indices.device, joint_valid_mask, mask)
            if bool((indices[~valid] != -1).any()):
                raise ValueError("Masked latent IDs must be -1 in every residual stage")
        output = torch.zeros((*shape, self.dim), dtype=torch.float32, device=indices.device)
        for stage_index, stage in enumerate(self.codebooks):
            local = indices[..., stage_index].long()
            if bool(((local[valid] < 0) | (local[valid] >= self.codebook_size)).any()):
                raise ValueError(f"Invalid local ID for RVQ stage {stage_index}; PQ offsets are unsupported")
            safe = torch.where(valid, local, 0)
            output = output + F.embedding(safe, stage.codebook).masked_fill(~valid[..., None], 0.0)
        return output

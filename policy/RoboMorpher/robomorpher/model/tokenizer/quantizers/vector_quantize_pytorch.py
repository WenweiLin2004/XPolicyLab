"""EAT product-VQ adapter for vector-quantize-pytorch 1.31.1.

Each group is an independent, single-head raw-L2 EMA quantizer. There are no
new projections or trainable codebook parameters. This is a new checkpoint
backend; quant_ema state dictionaries must keep their original backend.

Training uses immediate EMA or the library's explicit accumulation boundary.
Intermediate forwards pass accum_ema_update=True; the boundary forward passes
False and the library updates EMA before backward/optimizer execution. Optimizer
overflow skips do not roll this update back. Kmeans uses the first valid
microbatch; dead-code replacement uses only the current boundary microbatch.
All-padding global batches add no statistics; an empty boundary still consumes
any earlier pending statistics through the library. A locally empty rank is rejected on
every rank before entering the library's initialization/replacement collectives.
Inference/index APIs require initialized codebooks and never change their state.
"""
from __future__ import annotations

import math
from importlib.metadata import version
from typing import Optional

import torch
from torch import distributed as dist, nn
from torch.nn import functional as F
from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch.vector_quantize_pytorch import cdist as _library_cdist

BACKEND_VERSION = "1.31.1"


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


class VectorQuantizerEMAM(nn.Module):
    """Keep EAT's PQ shapes, group-offset IDs and four-value forward contract.

    ``vocab_size`` and ``vocab_width`` are totals across all groups. The training
    commitment value is the global valid-element mean. Its local gradient is
    scaled to compensate for DDP's subsequent averaging of parameter gradients.
    Consequently all ranks must enter distributed training ``forward`` together.
    Eval and index/lookup methods are rank-local and need no collectives, so an
    uneven evaluation loader cannot hang while updating metrics.
    """

    def __init__(
        self, vocab_size, vocab_width, beta=0.25, use_entropy_loss=False,
        entropy_temp=0.01, num_codebooks=16, ema_decay=0.8, kmeans_iters=100,
        dead_code_threshold=2.0, use_rotation_trick=True, *,
        sync_codebook: Optional[bool] = None, use_cosine_sim=False,
        stochastic_sample_codes=False, learnable_codebook=False,
        ema_update=True, ema_update_unit="microbatch_forward", **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(f"Unsupported VQP adapter options: {sorted(kwargs)}")
        if version("vector-quantize-pytorch") != BACKEND_VERSION:
            raise RuntimeError(f"EAT VQP backend requires vector-quantize-pytorch=={BACKEND_VERSION}")
        if use_entropy_loss or use_cosine_sim or stochastic_sample_codes:
            raise ValueError("EAT VQP backend supports raw L2, deterministic selection and no entropy loss only")
        if learnable_codebook or not ema_update:
            raise ValueError("EAT VQP backend requires EMA updates and a non-learnable codebook")
        if ema_update_unit == "optimizer_step":
            raise ValueError("optimizer_step is no longer supported; explicitly select accumulation_boundary for the official forward-boundary strategy")
        if ema_update_unit not in ("microbatch_forward", "accumulation_boundary"):
            raise ValueError("ema_update_unit must be 'microbatch_forward' or 'accumulation_boundary'")
        self.ema_update_unit = ema_update_unit
        self.vocab_size, self.vocab_width = int(vocab_size), int(vocab_width)
        self.num_codebooks = int(num_codebooks)
        if min(self.vocab_size, self.vocab_width, self.num_codebooks) <= 0:
            raise ValueError("Vocabulary size, width and number of groups must be positive")
        if self.vocab_size % self.num_codebooks or self.vocab_width % self.num_codebooks:
            raise ValueError("Total vocabulary size and width must be divisible by num_codebooks")
        self.group_size = self.vocab_size // self.num_codebooks
        self.group_width = self.vocab_width // self.num_codebooks
        self.beta, self.ema_decay = float(beta), float(ema_decay)
        self.dead_code_threshold = float(dead_code_threshold)
        self.kmeans_iters = int(kmeans_iters)
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if not 0 <= self.ema_decay <= 1 or self.kmeans_iters < 1:
            raise ValueError("EMA decay must be in [0,1] and kmeans_iters >= 1")
        if not math.isfinite(self.dead_code_threshold) or self.dead_code_threshold < 0:
            raise ValueError("dead_code_threshold must be finite and nonnegative")
        if not math.isfinite(float(entropy_temp)) or float(entropy_temp) <= 0:
            raise ValueError("entropy_temp must be finite and positive")
        self.use_rotation_trick = bool(use_rotation_trick)
        self.use_entropy_loss = False
        self.sync_codebook = _distributed() if sync_codebook is None else bool(sync_codebook)
        if self.sync_codebook != _distributed():
            raise RuntimeError("Construct the VQP adapter after distributed initialization; do not disable DDP codebook sync")
        self.codebooks = nn.ModuleList([
            VectorQuantize(
                dim=self.group_width, codebook_dim=self.group_width,
                codebook_size=self.group_size, heads=1, separate_codebook_per_head=False,
                kmeans_init=True, kmeans_iters=self.kmeans_iters,
                decay=self.ema_decay, ema_update=True, learnable_codebook=False,
                commitment_weight=self.beta, threshold_ema_dead_code=self.dead_code_threshold,
                use_cosine_sim=False, rotation_trick=self.use_rotation_trick,
                stochastic_sample_codes=False, sync_codebook=self.sync_codebook,
                sync_kmeans=True, return_zeros_for_masked_padding=True,
            ) for _ in range(self.num_codebooks)
        ])
        self.register_buffer("vocab_usage", torch.zeros(self.num_codebooks, self.group_size))
        self.register_buffer("vocab_usage_record_times", torch.zeros((), dtype=torch.long))

    def _apply(self, fn, recurse=True):
        self._require_no_pending("move or cast the quantizer")
        # Device moves are allowed, but model.half()/bfloat16() must not lower
        # codebook/EMA precision. Keep original FP32 values: converting to half
        # first and then back to float would irreversibly round the EMA state.
        modules = list(self.modules()) if recurse else [self]
        preserved = [(module, name, buffer.detach().float().clone())
                     for module in modules for name, buffer in module._buffers.items()
                     if buffer is not None and buffer.is_floating_point()]
        result = super()._apply(fn, recurse=recurse)
        for module, name, original in preserved:
            converted = module._buffers[name]
            module._buffers[name] = original.to(device=converted.device, dtype=torch.float32)
        return result

    @property
    def has_pending_ema(self):
        return any(
            group._codebook.cluster_size.grad is not None or group._codebook.embed_avg.grad is not None
            for group in self.codebooks
        )

    def _require_no_pending(self, operation):
        if self.has_pending_ema:
            raise RuntimeError(
                f"Cannot {operation} with pending EMA; complete an accumulation boundary "
                "or explicitly discard_pending_ema() for exception recovery first"
            )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self._require_no_pending("save checkpoint state")
        return super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._require_no_pending("load checkpoint state")
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @torch.no_grad()
    def discard_pending_ema(self):
        """Explicit exception recovery only; normal training never calls this.

        Call on every distributed rank before continuing. This discards raw
        pending statistics, not committed EMA or first-microbatch kmeans state.
        There is deliberately no optimizer-success/skip integration.
        """
        had_pending = self.has_pending_ema
        for group in self.codebooks:
            group._codebook.cluster_size.grad = None
            group._codebook.embed_avg.grad = None
        return had_pending

    def init_vocab(self, eini: float):
        """Compatibility entry point: valid training data initialize via kmeans.

        Pre-kmeans random initialization is intentionally unnecessary. Refuse to
        silently destroy an initialized table if an old caller invokes this late.
        """
        if any(bool(group._codebook.initted) for group in self.codebooks):
            raise RuntimeError("Cannot reinitialize a populated VQP codebook through init_vocab")
        if not math.isfinite(float(eini)):
            raise ValueError("eini must be finite")

    @staticmethod
    def _mask(shape, device, joint_valid_mask=None, mask=None):
        if joint_valid_mask is not None and mask is not None:
            raise ValueError("Pass joint_valid_mask or mask, not both")
        valid = joint_valid_mask if mask is None else mask
        if valid is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool:
            raise TypeError("The joint mask must be a boolean tensor")
        valid = valid.to(device=device)
        if valid.ndim == 1 and len(shape) == 2 and valid.shape[0] == shape[1]:
            valid = valid.unsqueeze(0).expand(shape)
        if tuple(valid.shape) != tuple(shape):
            raise ValueError(f"Mask shape {tuple(valid.shape)} does not match {tuple(shape)}")
        return valid

    def _features(self, features, joint_valid_mask=None, mask=None):
        if not isinstance(features, torch.Tensor) or not features.is_floating_point():
            raise TypeError("features must be a floating-point tensor")
        if features.ndim != 3 or features.shape[-1] != self.vocab_width:
            raise ValueError(f"features must have shape [B,J,{self.vocab_width}]")
        valid = self._mask(features.shape[:2], features.device, joint_valid_mask, mask)
        # masked_fill, rather than multiplication, also removes NaN/Inf padding.
        clean = features.float().masked_fill(~valid[..., None], 0.0)
        return clean, valid

    def _require_initialized(self):
        self._require_no_pending("evaluate or query codebook indices")
        if not all(bool(group._codebook.initted) for group in self.codebooks):
            raise RuntimeError("VQP codebooks are uninitialized; eval/index APIs cannot initialize from evaluation data")

    def _forward_status(self, clean, valid, accum_ema_update):
        if self.sync_codebook != _distributed():
            raise RuntimeError("Distributed state changed after VQP construction; rebuild the adapter in the intended process group")
        boundary_mode = self.ema_update_unit == "accumulation_boundary"
        valid_option = (isinstance(accum_ema_update, bool) if boundary_mode
                        else accum_ema_update is None or accum_ema_update is False)
        accumulate = accum_ema_update is True
        initialized = [int(bool(group._codebook.initted)) for group in self.codebooks]
        pending = [int(value is not None) for group in self.codebooks
                   for value in (group._codebook.cluster_size.grad, group._codebook.embed_avg.grad)]
        local_count = int(valid.sum())
        world = dist.get_world_size() if self.sync_codebook else 1
        status = torch.tensor(
            [local_count, int(local_count == 0), int(not bool(torch.isfinite(clean).all())),
             int(self.training), int(not valid_option), int(accumulate), int(boundary_mode),
             *initialized, *pending],
            dtype=torch.long, device=clean.device,
        )
        if self.sync_codebook:
            dist.all_reduce(status, op=dist.ReduceOp.SUM)
        global_count = int(status[0])
        if int(status[2]):
            raise ValueError("Non-finite valid VQ features on at least one rank")
        if int(status[3]) not in (0, world):
            raise RuntimeError("All ranks must use the same VQP train/eval mode")
        if int(status[4]):
            raise ValueError("accumulation_boundary training requires explicit bool accum_ema_update; microbatch_forward accepts only None/False")
        if int(status[5]) not in (0, world) or int(status[6]) not in (0, world):
            raise RuntimeError("All ranks must use the same accum_ema_update flag and EMA update unit")
        initialized = status[7:7 + self.num_codebooks]
        pending = status[7 + self.num_codebooks:]
        if any(int(flag) not in (0, world) for flag in initialized):
            raise RuntimeError("Codebook initialization flags disagree across ranks")
        if any(int(flag) not in (0, world) for flag in pending):
            raise RuntimeError("Pending EMA flags disagree across ranks")
        if bool(pending.any()) and not bool((pending == world).all()):
            raise RuntimeError("Incomplete pending EMA statistics across PQ groups; explicitly discard before recovery")
        if bool(pending.any()) and not bool((initialized == world).all()):
            raise RuntimeError("Pending EMA requires initialized codebooks")
        if global_count and int(status[1]):
            raise RuntimeError("Locally empty VQ rank in a nonempty global batch is not supported; all ranks must provide valid tokens")
        return local_count, global_count, world, accumulate

    def _lookup(self, clean, valid):
        """Pure lookup using exactly the pinned library's FP32 distance rule."""
        ids, vectors = [], []
        with torch.no_grad():
            for group, chunk in zip(self.codebooks, clean.split(self.group_width, dim=-1)):
                table = group.codebook.detach()
                local_ids = torch.full(valid.shape, -1, dtype=torch.long, device=clean.device)
                values = torch.zeros_like(chunk)
                if bool(valid.any()):
                    nearest = _library_cdist(chunk[valid].unsqueeze(0), table.unsqueeze(0)).argmin(-1)[0]
                    local_ids[valid] = nearest
                    values[valid] = F.embedding(nearest, table)
                ids.append(local_ids)
                vectors.append(values)
        return ids, vectors

    def _global_loss(self, local_loss, local_count, global_count, world):
        if not self.sync_codebook:
            return local_loss
        global_value = local_loss.detach().clone() * local_count
        dist.all_reduce(global_value, op=dist.ReduceOp.SUM)
        global_value /= global_count
        scaled_local = local_loss * (world * local_count / global_count)
        # Same logged scalar on every rank; correct global mean after DDP's
        # gradient averaging, without differentiating through an all-reduce.
        return scaled_local + (global_value - scaled_local.detach())

    def _usage(self, ids, valid, update):
        counts = torch.stack([torch.bincount(index[valid], minlength=self.group_size)
                              for index in ids]).float()
        if self.sync_codebook and update:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        probabilities = counts / counts.sum(-1, keepdim=True).clamp_min(1)
        usage = (probabilities > 0.01 / self.group_size).float().mean() * 100
        if update:
            with torch.no_grad():
                times = int(self.vocab_usage_record_times)
                if times == 0:
                    self.vocab_usage.copy_(probabilities)
                else:
                    rate = 0.1 if times < 100 else 0.01
                    self.vocab_usage.lerp_(probabilities, rate)
                self.vocab_usage_record_times.add_(1)
        return usage

    def forward(self, features, joint_valid_mask=None, *, mask=None,
                accum_ema_update: Optional[bool] = None):
        clean, valid = self._features(features, joint_valid_mask, mask)
        if self.training:
            local_count, global_count, world, accumulate = self._forward_status(clean, valid, accum_ema_update)
        else:
            # Do not synchronize evaluation: ranks may have different numbers
            # of batches, and rank-zero-only diagnostics must also remain safe.
            if accum_ema_update is not None and accum_ema_update is not False:
                raise ValueError("Evaluation accepts only accum_ema_update=None/False and never updates EMA")
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
                    # Consume earlier raw statistics through the official public
                    # forward, with zero current counts/sums. The library's empty
                    # commitment is NaN, so discard its outputs under no_grad.
                    # Every rank is empty: replacement has no current candidates
                    # and safely skips without entering distributed sampling.
                    with torch.no_grad():
                        for group, chunk in zip(self.codebooks, clean.split(self.group_width, dim=-1)):
                            group(chunk.detach(), mask=valid, accum_ema_update=False)
                # No pending data, or a non-boundary empty microbatch: no init,
                # no EMA update, and no change to the per-forward usage metric.
                return clean * 0.0, zero, zero, zero.detach()
            if self.training:
                vectors, ids, losses = [], [], []
                for group, chunk in zip(self.codebooks, clean.split(self.group_width, dim=-1)):
                    quantized, index, loss = group(
                        chunk, mask=valid,
                        accum_ema_update=accumulate,
                    )
                    vectors.append(quantized)
                    ids.append(index)
                    losses.append(loss.reshape(()))
                local_loss = torch.stack(losses).mean()
                quantized = torch.cat(vectors, dim=-1)
            else:
                ids, vectors = self._lookup(clean, valid)
                selected = torch.cat(vectors, dim=-1)
                local_loss = (selected[valid] - clean[valid]).square().mean() * self.beta
                # Preserve the eval STE when gradients are explicitly
                # requested, while every codebook/usage buffer stays read-only.
                quantized = selected + (clean - clean.detach())
            loss = (self._global_loss(local_loss, local_count, global_count, world)
                    if self.training else local_loss)
            usage = self._usage(ids, valid, update=self.training)
            return quantized, loss, zero, usage

    @torch.no_grad()
    def f_to_idx(self, features, joint_valid_mask=None, *, mask=None):
        clean, valid = self._features(features, joint_valid_mask, mask)
        self._require_initialized()
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("Non-finite valid VQ features")
        with torch.autocast(device_type=features.device.type, enabled=False):
            ids, vectors = self._lookup(clean, valid)
            global_ids = [torch.where(valid, index + g * self.group_size, -1)
                          for g, index in enumerate(ids)]
            return torch.stack(global_ids, dim=2), torch.stack(vectors, dim=2)

    @staticmethod
    def _integer_ids(indices):
        if not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("indices must be int32/int64 tensors")

    def idx_to_f(self, indices, joint_valid_mask=None, *, mask=None):
        self._integer_ids(indices)
        if indices.ndim != 3 or indices.shape[1] != self.num_codebooks:
            raise ValueError(f"indices must have shape [B,{self.num_codebooks},J]")
        self._require_initialized()
        shape = (indices.shape[0], indices.shape[2])
        if joint_valid_mask is None and mask is None:
            padding = indices == -1
            if bool((padding.any(1) != padding.all(1)).any()):
                raise ValueError("A padded joint must use -1 in every PQ group")
            valid = ~padding.all(1)
        else:
            valid = self._mask(shape, indices.device, joint_valid_mask, mask)
        values = []
        for g, group in enumerate(self.codebooks):
            local = indices[:, g].long() - g * self.group_size
            if bool(((local[valid] < 0) | (local[valid] >= self.group_size)).any()):
                raise ValueError(f"Invalid group-offset ID for PQ group {g}")
            safe = torch.where(valid, local, 0)
            vector = F.embedding(safe, group.codebook)
            values.append(vector.masked_fill(~valid[..., None], 0.0))
        return torch.cat(values, dim=-1)

    def idx_to_specifical_codebook_f(self, codebook_idx: int, indices,
                                    joint_valid_mask=None, *, mask=None):
        """Single-group API intentionally takes LOCAL IDs, with -1 for padding."""
        self._integer_ids(indices)
        self._require_initialized()
        if not 0 <= int(codebook_idx) < self.num_codebooks:
            raise ValueError("Invalid codebook_idx")
        if joint_valid_mask is None and mask is None:
            valid = indices != -1
        else:
            valid = self._mask(indices.shape, indices.device, joint_valid_mask, mask)
        if bool(((indices[valid] < 0) | (indices[valid] >= self.group_size)).any()):
            raise ValueError("Single-codebook lookup expects local IDs, not group-offset IDs")
        safe = torch.where(valid, indices.long(), 0)
        vector = F.embedding(safe, self.codebooks[int(codebook_idx)].codebook)
        return vector.masked_fill(~valid[..., None], 0.0)

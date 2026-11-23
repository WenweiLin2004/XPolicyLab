"""Lazy-loading single-step dataset for action tokenizer training.

Lazy loading (no full-dataset RAM load), adapted for zarr:
  - normalization stats are computed ONCE on the main process, cached to disk
    (atomic write), and every other rank loads the cache after a barrier
  - samples are read from disk on demand; each DataLoader worker caches only
    the current 4096-row block (~2.5MB), so RAM stays bounded
  - random single-row reads would require decompressing a whole zarr chunk per
    sample (~140ms); instead a block-shuffled batch sampler yields shuffled
    block order + shuffled rows within each block, so every disk read is a
    vectorized block read and batches stay near-i.i.d.
  - multi-rank sharding is handled by accelerate at batch level
    (BatchSamplerShard over our batch sampler).

Expected layout (produced by the RoboTwin and EmbodiBench converters):
    <data_dir>/index.json
    <data_dir>/data_chunk_*.zarr/data/{input (N,J,7), output (N,J,10), joint_valid_mask (N,J)}

Joint axis is truncated at load/stats time to ``vqvae_model.max_num_joints``.
If any sample has a valid joint beyond that index, loading raises ValueError.
New cache names include schema/scope/representation/method/J/train split identity.
New stats use training rows and valid joints only, shared across slots by default.
Frozen [J,D] checkpoint statistics remain readable without recomputation.
Joint angles (last channel of input/output) are always raw physical radians.
The removed atan2(sin, cos) preprocessing is never applied, so multi-turn
coordinates remain available to the tokenizer and policy.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right

import numpy as np
import torch
import zarr

from robomorpher.utils.geometry.rot6d import urdf_rpy_to_rot6d, urdf_rpy_to_rot6d_np
from robomorpher.model.tokenizer import normalize_model_config

from .masking import binary_joint_mask

KEYS = ("input", "output", "joint_valid_mask")


def _truncate_joint_arrays(x, y, mask, max_num_joints: int, *, where: str = ""):
    """Truncate joint axis to ``max_num_joints``; error if any valid joint beyond it.

    Accepts numpy or torch arrays with joint axis at -2 for x/y and -1 for mask
    (shapes (..., J, D) / (..., J)). If stored J < max_num_joints, raises —
    the model expects a fixed joint count and we do not pad here.
    """
    j_stored = int(mask.shape[-1])
    if j_stored < max_num_joints:
        raise ValueError(
            f"{where}: stored joints={j_stored} < max_num_joints={max_num_joints}; "
            "refusing to pad"
        )
    if j_stored > max_num_joints:
        overflow = mask[..., max_num_joints:]
        if isinstance(overflow, torch.Tensor):
            has_overflow = bool(overflow.any().item())
        else:
            has_overflow = bool(np.any(overflow))
        if has_overflow:
            # locate a few offending sample indices when mask is (N, J)
            detail = ""
            if mask.ndim >= 2:
                flat = overflow.reshape(overflow.shape[0], -1)
                row_hit = flat.any(dim=-1) if isinstance(flat, torch.Tensor) else flat.any(axis=-1)
                idxs = np.asarray(
                    row_hit.nonzero(as_tuple=False)[:5, 0]
                    if isinstance(row_hit, torch.Tensor)
                    else np.nonzero(row_hit)[0][:5]
                )
                detail = f" (e.g. row indices {idxs.tolist()})"
            raise ValueError(
                f"{where}: found valid joints beyond max_num_joints={max_num_joints} "
                f"(stored J={j_stored}){detail}"
            )
        x = x[..., :max_num_joints, :]
        y = y[..., :max_num_joints, :]
        mask = mask[..., :max_num_joints]
    return x, y, mask


def wrap_joint_angles(x, y, angle_wrap="none"):
    """Copy raw physical angles; wrapping is deliberately unsupported."""
    validate_angle_wrap(angle_wrap)
    if isinstance(x, torch.Tensor):
        return x.clone(), y.clone()
    return np.array(x, copy=True), np.array(y, copy=True)


# --------------------------------------------------------------------------- normalization
# Re-export versioned helpers for trainer/evaluator integrations.
from .normalization import (
    coerce_norm_stats, validate_norm_stats, load_norm_stats_cache,
    norm_stats_cache_payload, compute_norm_stats, load_or_compute_stats,
    dataset_norm_checkpoint, stats_request_metadata, stats_cache_filename,
    split_period, synchronize_norm_stage, validate_angle_wrap, validate_training_angle_wrap, norm_angle_wrap,
)


# --------------------------------------------------------------------------- dataset
class ZarrDataset(torch.utils.data.Dataset):
    """Lazy single-step dataset: __getitem__ serves one row from a per-worker block cache."""

    def __init__(
        self,
        data_dir: str,
        norm_stats: dict,
        split: str = "train",
        test_split: float = 0.01,
        normalization_method: str | None = None,
        urdf_rotation_6d: bool = False,
        max_num_joints: int | None = None,
        angle_wrap: str | None = None,
        input_representation: str = "joint",
    ):
        super().__init__()
        assert split in ("train", "test", "all")  # "all": every row (eval on full data)
        if max_num_joints is None:
            raise ValueError("max_num_joints is required (match vqvae_model.max_num_joints)")
        with open(os.path.join(data_dir, "index.json")) as f:
            index = json.load(f)

        self.data_dir = data_dir
        self.rows_per_chunk = int(index.get("rows_per_chunk", 4096))
        self.split = split
        self.normalization_method = str(normalization_method or norm_stats.get("normalization_method", "zscore"))
        self.urdf_rotation_6d = urdf_rotation_6d
        self.max_num_joints = int(max_num_joints)
        self.input_representation = str(input_representation).lower()
        if self.input_representation not in {"joint", "urdf_ee", "ee_only"}:
            raise ValueError("input_representation must be joint, urdf_ee or ee_only")

        # global block table over ALL rows (train and test interleave within blocks)
        self._shard_offsets = np.zeros(len(index["shards"]), dtype=np.int64)
        self._blocks: list[tuple[int, int, int]] = []  # (shard_idx, row_start, n_rows)
        offset = 0
        for shard_idx, shard in enumerate(index["shards"]):
            self._shard_offsets[shard_idx] = offset
            rows = int(shard["rows"])
            for start in range(0, rows, self.rows_per_chunk):
                self._blocks.append((shard_idx, start, min(self.rows_per_chunk, rows - start)))
            offset += rows
        self._shard_paths = [s["path"] for s in index["shards"]]
        self.total_rows = offset

        # deterministic sample-level split: every `period`-th global row -> test
        self.period = split_period(test_split)
        self.num_test = (self.total_rows + self.period - 1) // self.period if self.period else 0
        self.num_train = self.total_rows - self.num_test

        self.norm_stats = coerce_norm_stats(norm_stats, normalization_method=self.normalization_method)
        self.norm_scope = validate_norm_stats(
            self.norm_stats, max_num_joints=self.max_num_joints,
            urdf_rotation_6d=self.urdf_rotation_6d, normalization_method=self.normalization_method,
            angle_wrap=angle_wrap, input_representation=self.input_representation,
        )
        self.norm_metadata = self.norm_stats.get("metadata")
        self.angle_wrap = norm_angle_wrap(self.norm_stats)

        # Schema2 shared [D] broadcasts over every joint; old [J,D] is preserved.
        def _stat(key: str) -> torch.Tensor:
            return self.norm_stats[key]

        self.x_mean = _stat("x_mean")
        self.x_std = _stat("x_std")
        self.y_mean = _stat("y_mean")
        self.y_std = _stat("y_std")
        self.x_min = _stat("x_min")
        self.x_max = _stat("x_max")
        self.y_min = _stat("y_min")
        self.y_max = _stat("y_max")

        # q99 is a SEPARATE method from minmax: it uses the (q01, q99) quantile
        # range stored under x/y_q01/q99 keys; full-range x/y_min/max stay untouched.
        self.quantile = _parse_quantile(self.normalization_method)
        if self.quantile is not None:
            self.x_q01 = _stat("x_q01")
            self.x_q99 = _stat("x_q99")
            self.y_q01 = _stat("y_q01")
            self.y_q99 = _stat("y_q99")

        # per-worker state: zarr handles + single cached block (shard_idx, row_start)
        self._worker_arrays: dict[int, dict] | None = None
        self._block_key: tuple[int, int] | None = None
        self._block_data: dict[str, np.ndarray] | None = None

    # ---------------- split index arithmetic (O(1), no materialized index list)
    def train_idx_to_global_row(self, idx: int) -> int:
        if self.period == 0:
            return idx
        block, within = divmod(idx, self.period - 1)
        return block * self.period + within + 1

    def global_row_to_train_idx(self, row: np.ndarray) -> np.ndarray:
        if self.period == 0:
            return row
        # rows must be non-test rows (row % period != 0)
        return row - (row + self.period - 1) // self.period

    def __len__(self) -> int:
        if self.split == "train":
            return self.num_train
        if self.split == "test":
            return self.num_test
        return self.total_rows

    def blocks_for_epoch(self):
        """[(block_global_row_start, global_rows ndarray of this split)] per block."""
        out = []
        for shard_idx, start, n in self._blocks:
            g0 = self._shard_offsets[shard_idx] + start
            rows = np.arange(g0, g0 + n)
            if self.split == "all" or (self.split == "train" and self.period == 0):
                keep = np.ones(n, dtype=bool)
            elif self.period == 0:
                keep = np.zeros(n, dtype=bool)
            else:
                keep = rows % self.period == 0 if self.split == "test" else rows % self.period != 0
            out.append((shard_idx, start, rows[keep]))
        return out

    # ---------------- lazy reads with per-worker caching
    def _arrays(self, shard_idx: int) -> dict:
        if self._worker_arrays is None:
            self._worker_arrays = {}
        arrays = self._worker_arrays.get(shard_idx)
        if arrays is None:
            group = zarr.open_group(os.path.join(self.data_dir, self._shard_paths[shard_idx]), mode="r")
            arrays = {k: group["data"][k] for k in KEYS}
            self._worker_arrays[shard_idx] = arrays
        return arrays

    def _block(self, shard_idx: int, row_start: int) -> dict:
        if self._block_key != (shard_idx, row_start):
            arrays = self._arrays(shard_idx)
            stop = row_start + self.rows_per_chunk
            x = np.asarray(arrays["input"][row_start:stop])
            y = np.asarray(arrays["output"][row_start:stop])
            m = binary_joint_mask(
                arrays["joint_valid_mask"][row_start:stop],
                where=f"dataset block shard={shard_idx} row_start={row_start}",
            )
            x, y, m = _truncate_joint_arrays(
                x, y, m, self.max_num_joints,
                where=f"dataset block shard={shard_idx} row_start={row_start}",
            )
            if self.input_representation == "urdf_ee":
                x = np.concatenate([x[..., :6], y[..., :9]], axis=-1)
                y = y[..., :9]
            elif self.input_representation == "ee_only":
                x = y[..., :9]
                y = y[..., :9]
            self._block_data = {"input": x, "output": y, "joint_valid_mask": m}
            self._block_key = (shard_idx, row_start)
        return self._block_data

    def _normalize(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalization_method == "zscore":
            x = (x - self.x_mean) / self.x_std
            y = (y - self.y_mean) / self.y_std
        elif self.normalization_method == "minmax":
            # full-range minmax -> [-1, 1]
            x = (x - self.x_min) / (self.x_max - self.x_min + 1e-6) * 2.0 - 1.0
            y = (y - self.y_min) / (self.y_max - self.y_min + 1e-6) * 2.0 - 1.0
        else:
            # q99: (q01, q99) range -> [-1, 1], outliers clamp
            x = ((x - self.x_q01) / (self.x_q99 - self.x_q01 + 1e-6) * 2.0 - 1.0).clamp(-1.0, 1.0)
            y = ((y - self.y_q01) / (self.y_q99 - self.y_q01 + 1e-6) * 2.0 - 1.0).clamp(-1.0, 1.0)
        return x, y

    def __getitem__(self, idx: int):
        if self.split == "test":
            global_row = idx * self.period
        elif self.split == "all":
            global_row = idx
        else:
            global_row = self.train_idx_to_global_row(idx)
        shard_idx = int(bisect_right(self._shard_offsets, global_row) - 1)
        local = int(global_row - self._shard_offsets[shard_idx])
        row_start = local - local % self.rows_per_chunk
        block = self._block(shard_idx, row_start)
        r = local - row_start
        x = torch.from_numpy(block["input"][r]).float()
        y = torch.from_numpy(block["output"][r]).float()
        mask = torch.from_numpy(block["joint_valid_mask"][r]).bool()  # (J,)
        # Padding is not a physical feature, including NaN/Inf sentinel values.
        x = x.masked_fill(~mask[..., None], 0.0)
        y = y.masked_fill(~mask[..., None], 0.0)
        if self.urdf_rotation_6d:
            # (J,7) -> (J,10); zero padding joints AFTER conversion (identity
            # rot6d is non-zero), matching the old full-load dataset behavior
            x = urdf_rpy_to_rot6d(x, mask)
        x, y = wrap_joint_angles(x, y, angle_wrap=self.angle_wrap)
        x, y = self._normalize(x, y)
        # canonical zero padding AFTER normalization too: degenerate stats
        # (e.g. never-valid joints) must not leak garbage/NaN into the model
        return x.masked_fill(~mask[..., None], 0.0), y.masked_fill(~mask[..., None], 0.0), mask


# --------------------------------------------------------------------------- samplers
class BlockShuffleBatchSampler(torch.utils.data.Sampler):
    """Yields batches of dataset indices grouped by zarr block.

    Shuffles block order and rows within each block per epoch, so batches are
    near-i.i.d. while every disk read is a vectorized block read.
    """

    def __init__(self, dataset: ZarrDataset, batch_size: int, seed: int = 42, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        # keep only per-block row counts; index arrays are built on the fly per block
        self._blocks = [(s, st, n) for s, st, n in dataset._blocks]
        period = dataset.period
        self._counts = []
        for shard_idx, start, n in self._blocks:
            g0 = int(dataset._shard_offsets[shard_idx]) + start
            # multiples of period in [g0, g0+n); period=0 when test_split=0
            n_test = (g0 + n - 1) // period - (g0 - 1) // period if period else 0
            if dataset.split == "all":
                self._counts.append(n)
            else:
                self._counts.append(n_test if dataset.split == "test" else n - n_test)
        total = sum(self._counts)
        self._num_batches = total // batch_size if drop_last else (total + batch_size - 1) // batch_size

    def _block_indices(self, block_idx: int) -> np.ndarray:
        """Dataset indices (this split) for one zarr block, computed on the fly."""
        ds = self.dataset
        shard_idx, start, n = self._blocks[block_idx]
        g0 = int(ds._shard_offsets[shard_idx]) + start
        rows = np.arange(g0, g0 + n, dtype=np.int64)
        if ds.split == "all":
            return rows  # dataset index == global row
        if ds.period == 0:
            return rows if ds.split == "train" else np.empty(0, dtype=np.int64)
        if ds.split == "test":
            return (rows[rows % ds.period == 0] // ds.period).astype(np.int64)
        rows = rows[rows % ds.period != 0]
        return ds.global_row_to_train_idx(rows).astype(np.int64)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _iter_batches(self, order: np.ndarray, permute_rows: bool, rng=None):
        """Chunk the (block-shuffled) row stream into batches.

        Batches may span block boundaries, so any batch_size works regardless of
        rows-per-block; workers still hit their block cache ~99% of the time.
        """
        buf = np.empty(0, dtype=np.int64)
        for b in order:
            rows = self._block_indices(int(b))
            if permute_rows:
                rows = rows[rng.permutation(len(rows))]
            buf = np.concatenate([buf, rows]) if len(buf) else rows
            while len(buf) >= self.batch_size:
                yield buf[: self.batch_size].tolist()
                buf = buf[self.batch_size :]
        if not self.drop_last and len(buf) > 0:
            yield buf.tolist()

    def __iter__(self):
        # Like DistributedSampler, set_epoch() owns the epoch clock. DataLoader
        # may construct an unused sampler iterator during worker startup; an
        # implicit increment here made cold and persistent iterators disagree.
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self._blocks))
        return self._iter_batches(order, permute_rows=True, rng=rng)

    def __len__(self):
        return self._num_batches


class SequentialBlockBatchSampler(BlockShuffleBatchSampler):
    """Same block grouping, but deterministic order (for the test split)."""

    def __iter__(self):
        order = np.arange(len(self._blocks))
        return self._iter_batches(order, permute_rows=False)


# --------------------------------------------------------------------------- module
def _parse_quantile(method: str) -> float | None:
    """'q99' -> 0.99 (quantile minmax); 'minmax'/'zscore' -> None."""
    if method in ("minmax", "zscore"):
        return None
    if method == "q99":
        return 0.99
    raise ValueError(f"unknown normalization_method: {method!r} (expected zscore|minmax|q99)")


def create_dataloaders(cfg, accelerator, norm_stats=None, normalization_method=None):
    """Build train/test loaders with rank-0-once stats; returns loaders + norm stats.

    If ``norm_stats`` is provided (e.g. frozen stats from a resume checkpoint),
    skip the disk cache / recompute path and use those tensors as-is.
    """
    data_dir = str(cfg.train_data_dir)
    if normalization_method is None:
        normalization_method = str(cfg.get("normalization_method", "minmax"))
    else:
        normalization_method = str(normalization_method)
    test_split = float(cfg.get("test_split", 0.01))
    angle_wrap = validate_training_angle_wrap(cfg.get("joint_angle_wrap"))
    vm = normalize_model_config(cfg.vqvae_model)
    urdf_rotation_6d = bool(vm["urdf_rotation_6d"])
    max_num_joints = int(vm["max_num_joints"])
    input_representation = ("urdf_ee" if vm.get("state_input_mode") == "ee" and vm.get("urdf_injection_enabled")
                            else "ee_only" if vm.get("state_input_mode") == "ee" else "joint")
    quantile = _parse_quantile(normalization_method)

    fresh_stats = norm_stats is None
    stats_scope = str(cfg.get("normalization_scope", "shared_feature"))
    if fresh_stats:
        norm_stats = load_or_compute_stats(
            data_dir, accelerator,
            workers=int(cfg.get("stats_workers", 16)),
            urdf_rotation_6d=urdf_rotation_6d,
            max_num_joints=max_num_joints,
            quantile=quantile,
            test_split=test_split,
            normalization_method=normalization_method,
            stats_scope=stats_scope,
            angle_wrap=angle_wrap, input_representation=input_representation,
        )
    elif accelerator.is_main_process:
        print(
            f"[stats] using provided norm_stats "
            f"(method={normalization_method}, keys={sorted(k for k in norm_stats if k != 'normalization_method')})"
        )

    train_dataset, test_dataset, error, training_identity = None, None, None, None
    try:
        train_dataset = ZarrDataset(
            data_dir, norm_stats, split="train",
            test_split=test_split, normalization_method=normalization_method,
            urdf_rotation_6d=urdf_rotation_6d,
            max_num_joints=max_num_joints,
            angle_wrap=angle_wrap, input_representation=input_representation,
        )
        test_dataset = ZarrDataset(
            data_dir, norm_stats, split="test",
            test_split=test_split, normalization_method=normalization_method,
            urdf_rotation_6d=urdf_rotation_6d,
            max_num_joints=max_num_joints,
            angle_wrap=angle_wrap, input_representation=input_representation,
        )
        if train_dataset.norm_metadata is not None:
            frozen_scope = train_dataset.norm_metadata["scope"]
            requested_scope = cfg.get("normalization_scope")
            if requested_scope is not None and str(requested_scope) != frozen_scope:
                raise ValueError("Requested normalization_scope differs from frozen stats; use a new run/explicit migration")
            expected = stats_request_metadata(
                data_dir, normalization_method=normalization_method, stats_scope=frozen_scope,
                urdf_rotation_6d=urdf_rotation_6d, max_num_joints=max_num_joints,
                test_split=test_split, quantile=quantile, angle_wrap=angle_wrap, input_representation=input_representation,
            )
            try:
                validate_norm_stats(train_dataset.norm_stats, max_num_joints=max_num_joints,
                                    urdf_rotation_6d=urdf_rotation_6d,
                                    normalization_method=normalization_method, expected_metadata=expected,
                                    angle_wrap=angle_wrap, input_representation=input_representation)
            except ValueError as exc:
                raise ValueError("Training data/split differs from frozen normalization identity; "
                                 "use a new run or explicit migration instead of strict resume") from exc
            training_identity = expected
        elif accelerator.is_main_process:
            print("[stats] warning: statistics have no train-split metadata; "
                  "identity cannot be verified. Keeping the original frozen tensors unchanged.")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    synchronize_norm_stage("training dataset construction", error, training_identity)
    if accelerator.is_main_process:
        print(f"Dataset split: train={len(train_dataset)}, test={len(test_dataset)}")

    num_workers = int(cfg.get("num_workers", 8))
    # DataLoader draws a worker base_seed when constructing an iterator. Keep
    # those draws separate from model/codebook RNG, especially when persistent
    # workers are recreated by a resume. Zarr reads are deterministic; sample
    # order remains controlled by the epoch-seeded batch sampler below.
    loader_seed = int(cfg.get("seed", 42)) + 2 * int(getattr(accelerator, "process_index", 0))
    train_generator = torch.Generator().manual_seed(loader_seed)
    test_generator = torch.Generator().manual_seed(loader_seed + 1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=BlockShuffleBatchSampler(
            train_dataset, cfg.batch_size, seed=int(cfg.get("seed", 42)), drop_last=True
        ),
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=train_generator,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=SequentialBlockBatchSampler(test_dataset, cfg.batch_size, drop_last=False),
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=test_generator,
    )
    return train_dataset, test_dataset, train_loader, test_loader

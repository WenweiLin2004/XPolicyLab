"""Host wall-time accounting at existing synchronized training boundaries.

No explicit CUDA synchronization is introduced. Data waiting and compute can
include overlapping device work; their sum is the measured training duration.
Logging, checkpoints, evaluation and initial model setup stay outside it.
"""
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist

class TrainingPerformance:
    def __init__(self, *, warmup_updates=10, clock=perf_counter):
        self.clock = clock
        self.warmup_updates = int(warmup_updates)
        self.completed_updates = 0
        self.last_update = None
        self._fetch_started = self._compute_started = None
        self._update_started = self._last_finished = None
        self._update_data = self._update_compute = 0.0
        self._update_samples = 0
        self._window = []

    def start_fetch(self):
        # Called only after logging/checkpoint/evaluation/replayed batches end.
        self._fetch_started = self.clock()
        if self._update_samples == 0:
            self._update_started = self._fetch_started

    def batch_ready(self, samples):
        now = self.clock()
        if self._fetch_started is None:
            raise RuntimeError("start_fetch must precede batch_ready")
        self._update_data += now - self._fetch_started
        self._compute_started = now
        self._update_samples += int(samples)

    def finish_microbatch(self):
        if self._compute_started is None:
            raise RuntimeError("batch_ready must precede finish_microbatch")
        self._last_finished = self.clock()
        self._update_compute += self._last_finished - self._compute_started
        self._compute_started = None

    def complete_update(self, step):
        self.last_update = {"step": int(step), "data_times": self._update_data,
                            "model_times": self._update_compute}
        self.completed_updates += 1
        if self.completed_updates > self.warmup_updates:
            self._window.append((int(step), self._update_data, self._update_compute, self._update_samples,
                                 self._update_started, self._last_finished))
        self._update_data = self._update_compute = 0.0
        self._update_samples = 0

    def take_window(self):
        if not self._window:
            return None
        rows, self._window = self._window, []
        return dict(first_step=rows[0][0], last_step=rows[-1][0], updates=len(rows),
                    data_wait_s=sum(row[1] for row in rows),
                    compute_and_sync_s=sum(row[2] for row in rows),
                    samples=sum(row[3] for row in rows),
                    wall_elapsed_s=rows[-1][5] - rows[0][4])


def log_training_performance(timer, *, output, device, rank, world, step, epoch):
    """All ranks call on identical log steps; rank zero writes a separate file.

    Time ends after the trainer's existing loss.item()/float(grad) reads. Thus
    steady-state sec/update includes loading, forward, backward, optimizer and
    normal communication, but excludes console/W&B/checkpoint overhead. This is
    not end-to-end job throughput or a CUDA-kernel-only benchmark.
    """
    local = timer.take_window()
    if local is None:
        return None
    keys = ("first_step", "last_step", "updates", "data_wait_s", "compute_and_sync_s", "samples", "wall_elapsed_s")
    values = torch.tensor([local[key] for key in keys], dtype=torch.float64, device=device)
    if world > 1:
        gathered = [torch.empty_like(values) for _ in range(world)]
        dist.all_gather(gathered, values)
    else:
        gathered = [values]
    ranks = []
    for index, tensor in enumerate(gathered):
        row = dict(zip(keys, tensor.cpu().tolist()))
        for key in ("first_step", "last_step", "updates", "samples"):
            row[key] = int(row[key])
        row["rank"] = index
        row["training_s"] = row["data_wait_s"] + row["compute_and_sync_s"]
        ranks.append(row)
    if len({(row["first_step"], row["last_step"], row["updates"]) for row in ranks}) != 1:
        raise RuntimeError("Performance windows disagree across distributed ranks")
    slowest_seconds = max(row["training_s"] for row in ranks)
    updates = ranks[0]["updates"]
    record = dict(global_step=step, epoch=epoch, first_step=ranks[0]["first_step"],
                  last_step=ranks[0]["last_step"], updates=updates,
                  process_warmup_updates=timer.warmup_updates,
                  rank0_training_s=ranks[0]["training_s"], max_rank_training_s=slowest_seconds,
                  rank0_wall_elapsed_s=ranks[0]["wall_elapsed_s"],
                  max_rank_wall_elapsed_s=max(row["wall_elapsed_s"] for row in ranks),
                  rank0_wall_seconds_per_update=ranks[0]["wall_elapsed_s"] / updates,
                  seconds_per_update=slowest_seconds / updates,
                  updates_per_second=updates / slowest_seconds if slowest_seconds > 0 else None,
                  actual_global_samples=sum(row["samples"] for row in ranks),
                  global_samples_per_second=sum(row["samples"] for row in ranks) / slowest_seconds
                      if slowest_seconds > 0 else None,
                  timing_scope="training_only_excludes_setup_logging_checkpoint_eval",
                  wall_timing_scope="inclusive_between_first_fetch_and_last_update_completion",
                  ranks=ranks)
    if rank == 0:
        with (Path(output) / "performance.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    return record

"""Detached, count-weighted diagnostics for tokenizer reconstruction.

Reconstruction diagnostics accumulate error sums and valid joint counts.
Split EE mode stores the explicitly supplied xyz+rotation objective; reference
mode retains EE9 MSE. XYZ/rot6D MSE remain internal comparison diagnostics.
Angle and public position metrics follow the selected L1/L2 objective; an
additional angle_mse diagnostic is retained when joint L1 is selected.
Scalar losses intentionally retain the trainer's rank/microbatch mean
semantics; this module never changes the tensors used for backpropagation.
"""

from collections.abc import Mapping

import torch

# Keep this diagnostics module loadable by file path without importing the
# trainer/model package (the standalone CPU distributed oracle relies on it).
def _is_split_ee_loss(mode):
    return mode in {"xyz_mse_rot_geodesic", "xyz_rot_geodesic"}


def _regression_loss_type(loss_type):
    name = str(loss_type).strip().lower()
    aliases = {"l1": "l1", "mae": "l1", "l2": "l2", "mse": "l2"}
    if name not in aliases:
        raise ValueError(f"Unsupported regression metric type: {loss_type!r}; expected l1/l2 (mae/mse)")
    return aliases[name]


SCALAR_NAMES = (
    "total_loss", "ee_loss", "joint_loss", "recon_loss", "vq_loss",
    "entropy_loss", "code_usages",
)


class TokenizerMetricAccumulator:
    """Accumulate compatible FP64 packed states on the prediction device.

    Packed fields are J valid counts, J error sums for each configured diagnostic,
    training scalar sums (optionally including pos/rot), then one scalar-observation
    count. The explicit
    ``include_fk=True`` evaluation mode adds an FK SSE field and appends an
    ``fk_loss`` scalar; default training states contain neither. Only batches
    supplying the complete instance ``scalar_names`` mapping contribute to that
    final count. Per-joint observations are counted even without scalars.
    """

    SCALAR_NAMES = SCALAR_NAMES

    def __init__(self, num_joints, ee_dim, joint_dim, device, include_fk=False, ee_loss_type="mse", record_components=False, xyz_loss_type="l2", joint_loss_type="l2"):
        self.num_joints = int(num_joints)
        self.ee_dim = int(ee_dim)
        self.joint_dim = int(joint_dim)
        self.include_fk = bool(include_fk)
        self.ee_loss_type = str(ee_loss_type).strip().lower()
        self.xyz_loss_type = _regression_loss_type(xyz_loss_type)
        self.joint_loss_type = _regression_loss_type(joint_loss_type)
        if self.ee_loss_type != "mse" and not _is_split_ee_loss(self.ee_loss_type):
            raise ValueError("Unknown EE metric loss type")
        if self.ee_loss_type == "mse" and self.xyz_loss_type != "l2":
            raise ValueError("XYZ L1 metrics require split EE loss mode")
        if _is_split_ee_loss(self.ee_loss_type) and self.ee_dim not in (0, 9):
            raise ValueError("Split EE loss metrics require ee_dim=9")
        self.record_components = bool(record_components)
        if self.record_components and self.ee_dim != 9:
            raise ValueError("Position/rotation loss logging requires ee_dim=9")
        self.scalar_names = (SCALAR_NAMES + (("pos_loss", "rot_loss") if self.record_components else ())
                             + (("fk_loss",) if self.include_fk else ()))
        if self.num_joints <= 0 or self.joint_dim < 0 or self.ee_dim < 0:
            raise ValueError("joint dimensions must be positive; ee_dim may be zero")
        self.target_ee_dim = 9 if self.ee_dim == 0 else self.ee_dim
        self._dims = ({"ee": self.ee_dim} if self.ee_dim else {})
        self._dims["angle"] = self.joint_dim
        if self.ee_dim == 9:
            self._dims.update(ee_xyz=3, ee_rot6d=6)
            if self.record_components and _is_split_ee_loss(self.ee_loss_type):
                self._dims["ee_rotation"] = 1
            if self.xyz_loss_type == "l1":
                self._dims["ee_position"] = 3
        if self.joint_loss_type == "l1":
            self._dims["angle_mse"] = self.joint_dim
        if self.include_fk:
            self._dims["fk_ee"] = self.target_ee_dim
        self._count = slice(0, self.num_joints)
        cursor = self.num_joints
        # Historical attribute name retained; selected L1 fields store SAE.
        self._sse = {}
        for name in self._dims:
            self._sse[name] = slice(cursor, cursor + self.num_joints)
            cursor += self.num_joints
        self._scalars = slice(cursor, cursor + len(self.scalar_names))
        self._scalar_count = cursor + len(self.scalar_names)
        self.state = torch.zeros(self._scalar_count + 1, device=device, dtype=torch.float64)

    def _check_state(self, packed):
        if not isinstance(packed, torch.Tensor) or packed.shape != self.state.shape:
            shape = getattr(packed, "shape", None)
            raise ValueError(f"Expected packed state shape {tuple(self.state.shape)}, got {shape}")
        return packed.detach().to(device=self.state.device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, ee_pred, joint_pred, target, mask, *, fk_per_joint=None, ee_per_joint=None, rot_per_joint=None, scalars=None):
        """Add one local microbatch and return its independent detached state.

        Padding is cleared before subtraction/squaring, so padded NaN/Inf cannot
        enter metrics. Nonfinite values at valid joints remain visible.
        In FK-enabled evaluation, ``fk_per_joint`` is the existing masked FK
        MSE per slot, not a training loss. ``scalars`` must match this instance's
        ``scalar_names`` exactly; training mappings cannot contain ``fk_loss``.
        """
        if ee_pred.ndim != 3 or ee_pred.shape[1:] != (self.num_joints, self.ee_dim):
            raise ValueError("ee_pred must have shape [B, num_joints, ee_dim]")
        batch = ee_pred.shape[0]
        if joint_pred.shape != (batch, self.num_joints, self.joint_dim):
            raise ValueError("joint_pred must have shape [B, num_joints, joint_dim]")
        if target.shape != (batch, self.num_joints, self.target_ee_dim + self.joint_dim):
            raise ValueError("target must contain the EE channels followed by the joint channels")
        if ee_pred.device != self.state.device or joint_pred.device != self.state.device:
            raise ValueError("Accumulator and predictions must be on the same device")
        valid = mask.detach().to(device=self.state.device, dtype=torch.bool)
        if valid.ndim == 1 and valid.shape[0] == self.num_joints:
            valid = valid.unsqueeze(0).expand(batch, -1)
        if valid.shape != (batch, self.num_joints):
            raise ValueError("mask must have shape [B, num_joints] or [num_joints]")
        if scalars is not None:
            if not isinstance(scalars, Mapping) or set(scalars) != set(self.scalar_names):
                raise ValueError(f"scalars must supply exactly {self.scalar_names}")

        def clean(tensor):
            return tensor.detach().to(device=self.state.device, dtype=torch.float64).masked_fill(
                ~valid[..., None], 0.0
            )

        target_clean = clean(target)
        ee_difference = clean(ee_pred) - target_clean[..., :self.ee_dim]
        joint_difference = clean(joint_pred) - target_clean[..., self.target_ee_dim:]
        ee_squared = ee_difference.square()
        joint_squared = joint_difference.square()
        joint_error = joint_difference.abs() if self.joint_loss_type == "l1" else joint_squared
        packed = torch.zeros_like(self.state)
        counts = valid.sum(dim=0, dtype=torch.float64)
        packed[self._count] = counts
        if self.ee_dim == 0:
            if ee_per_joint is not None or rot_per_joint is not None:
                raise ValueError("angle-only metrics must not receive EE or rotation objectives")
        elif self.ee_loss_type == "mse":
            if ee_per_joint is not None:
                raise ValueError("Explicit EE objective requires split EE metric mode")
            packed[self._sse["ee"]] = ee_squared.sum(dim=(0, 2))
        else:
            if not isinstance(ee_per_joint, torch.Tensor) or ee_per_joint.shape != (self.num_joints,):
                raise ValueError("Split EE metrics require the actual per-slot reconstruction objective")
            means = ee_per_joint.detach().to(device=self.state.device, dtype=torch.float64)
            means = means.masked_fill(counts == 0, 0.0)
            # Preserve the packed layout. Dividing by count*ee_dim below
            # recovers the objective; this field is not an EE9 SSE in split mode.
            packed[self._sse["ee"]] = means * counts * self.ee_dim
        packed[self._sse["angle"]] = joint_error.sum(dim=(0, 2))
        if "angle_mse" in self._dims:
            packed[self._sse["angle_mse"]] = joint_squared.sum(dim=(0, 2))
        if self.ee_dim == 9:
            packed[self._sse["ee_xyz"]] = ee_squared[..., :3].sum(dim=(0, 2))
            packed[self._sse["ee_rot6d"]] = ee_squared[..., 3:9].sum(dim=(0, 2))
            if "ee_position" in self._dims:
                packed[self._sse["ee_position"]] = ee_difference[..., :3].abs().sum(dim=(0, 2))
        if "ee_rotation" in self._dims:
            if not isinstance(rot_per_joint, torch.Tensor) or rot_per_joint.shape != (self.num_joints,):
                raise ValueError("Split rotation metrics require the actual per-slot theta/pi")
            rotation = rot_per_joint.detach().to(device=self.state.device, dtype=torch.float64)
            # Already a per-valid-joint angle mean: no division by 6 or 9.
            packed[self._sse["ee_rotation"]] = rotation.masked_fill(counts == 0, 0.0) * counts
        if self.include_fk:
            if not isinstance(fk_per_joint, torch.Tensor) or fk_per_joint.shape != (self.num_joints,):
                raise ValueError("include_fk=True requires fk_per_joint with shape [num_joints]")
            fk_mean = fk_per_joint.detach().to(device=self.state.device, dtype=torch.float64)
            fk_mean = fk_mean.masked_fill(counts == 0, 0.0)
            packed[self._sse["fk_ee"]] = fk_mean * counts * self.target_ee_dim
        if scalars is not None:
            values = []
            for name in self.scalar_names:
                value = scalars[name]
                if isinstance(value, torch.Tensor):
                    value = value.detach()
                value = torch.as_tensor(value, device=self.state.device, dtype=torch.float64)
                if value.numel() != 1:
                    raise ValueError(f"scalars[{name!r}] must be a scalar")
                values.append(value.reshape(()))
            packed[self._scalars] = torch.stack(values)
            packed[self._scalar_count] = 1.0
        self.add(packed)
        return packed

    @torch.no_grad()
    def add(self, batch_stats):
        """Add a compatible batch state without retaining its computation graph."""
        self.state.add_(self._check_state(batch_stats))

    @torch.no_grad()
    def reset(self):
        self.state.zero_()

    @torch.no_grad()
    def synchronize(self, accelerator):
        """All ranks must call, including ranks with zero valid observations."""
        reduced = accelerator.reduce(self.state.detach().clone(), reduction="sum")
        return self._check_state(reduced)

    @torch.no_grad()
    def scalar_means(self, reduced_state):
        """Return unprefixed rank/microbatch objective means, or {} if absent."""
        # Formatting runs only on the main process. Copy the small packed
        # vector once instead of synchronizing CUDA for every scalar .item().
        packed = self._check_state(reduced_state).cpu()
        count = packed[self._scalar_count].item()
        if count == 0:
            return {}
        means = packed[self._scalars] / count
        return {name: means[i].item() for i, name in enumerate(self.scalar_names)}

    @torch.no_grad()
    def diagnostics_dict(self, prefix, reduced_state):
        """Format detailed internal diagnostics; not published to WandB.

        Retained for internal validation; training uses training_per_joint_dict.
        This does not modify the accumulator or the tensors used for training.
        """
        packed = self._check_state(reduced_state).cpu()
        counts = packed[self._count]
        result = {}
        for joint in range(self.num_joints):
            count = counts[joint].item()
            result[f"{prefix}/joint_{joint}_valid_count"] = int(count)
            if count == 0:
                continue
            for name, dim in self._dims.items():
                if dim <= 0:
                    continue
                result[f"{prefix}/joint_{joint}_{name}_loss"] = (
                    packed[self._sse[name]][joint] / (count * dim)
                ).item()
        total_count = counts.sum().item()
        result[f"{prefix}/valid_joint_count"] = int(total_count)
        if total_count > 0:
            names = {}
            if self.joint_dim > 0:
                names["angle"] = "joint"
            if self.ee_dim:
                names["ee"] = "ee"
            if self.ee_dim == 9:
                names.update(ee_xyz="ee_xyz", ee_rot6d="ee_rot6d")
                if "ee_position" in self._dims:
                    names["ee_position"] = "pos"
            if "angle_mse" in self._dims:
                names["angle_mse"] = "joint_mse"
            for name, log_name in names.items():
                result[f"{prefix}/valid_{log_name}_loss"] = (
                    packed[self._sse[name]].sum() / (total_count * self._dims[name])
                ).item()
        return result

    @torch.no_grad()
    def training_wandb_dict(self, reduced_state):
        """Compact optimizer-step log; every value has the scalar objective reduction.

        The current pose9 trainer publishes exactly five global keys. Component
        losses are unweighted; total_loss is the actual objective including VQ.
        Code usage is the model's average across stages (0 during AE bypass).
        No extra collective, recomputation, or change to autograd is performed.
        """
        means = self.scalar_means(reduced_state)
        if not means:
            return {}
        result = {"angle_loss": means["joint_loss"],
                  "total_loss": means["total_loss"], "code_usage": means["code_usages"]}
        if self.record_components:
            result = {"pos_loss": means["pos_loss"], "angle_loss": means["joint_loss"],
                      "rot_loss": means["rot_loss"], "total_loss": means["total_loss"],
                      "code_usage": means["code_usages"]}
        return result

    @torch.no_grad()
    def training_per_joint_dict(self, prefix, reduced_state):
        """Log categories first (pos, rot, angle), then ascending joint index.

        Position/angle use the selected coordinate MAE/MSE (unweighted).
        Split rotation uses the actual physical theta/pi, never rot6D MSE.
        Entirely absent slots are omitted instead of being logged as zero.
        """
        if self.ee_dim == 0:
            packed = self._check_state(reduced_state).cpu()
            counts = packed[self._count]
            return {f"{prefix}/angle_loss_joint_{j}":
                    (packed[self._sse["angle"]][j] / (counts[j] * self.joint_dim)).item()
                    for j in range(self.num_joints) if counts[j].item() > 0}
        if _is_split_ee_loss(self.ee_loss_type) and not self.record_components:
            raise ValueError("Per-joint split rotation logging requires record_components=True")
        if self.ee_dim != 9:
            return self.wandb_dict(prefix, reduced_state)
        packed = self._check_state(reduced_state).cpu()
        counts = packed[self._count]
        rotation_field = "ee_rotation" if _is_split_ee_loss(self.ee_loss_type) else "ee_rot6d"
        position_field = "ee_position" if self.xyz_loss_type == "l1" else "ee_xyz"
        result = {}
        categories = [("pos", position_field), ("rot", rotation_field)]
        if self.joint_dim > 0:
            categories.append(("angle", "angle"))
        for category, field in categories:
            for joint in range(self.num_joints):
                count = counts[joint].item()
                if count == 0:
                    continue
                result[f"{prefix}/{category}_loss_joint_{joint}"] = (
                    packed[self._sse[field]][joint] / (count * self._dims[field])
                ).item()
        return result

    @torch.no_grad()
    def wandb_dict(self, prefix, reduced_state):
        """per-joint EE/angle formatter, retained for non-pose9 callers.

        Counts remain internal denominators. They and the optional FK/XYZ/
        rotation diagnostics are deliberately not part of the public payload.
        """
        packed = self._check_state(reduced_state).cpu()
        counts = packed[self._count]
        result = {}
        for joint in range(self.num_joints):
            count = counts[joint].item()
            if count == 0:
                continue
            names = ["ee"]
            if self.joint_dim > 0:
                names.append("angle")
            for name in names if self.ee_dim else ["angle"]:
                result[f"{prefix}/joint_{joint}_{name}_loss"] = (
                    packed[self._sse[name]][joint] / (count * self._dims[name])
                ).item()
        return result

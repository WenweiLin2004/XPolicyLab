#!/usr/bin/env python3
"""Read-only physical metrics for one EAT tokenizer checkpoint.

This intentionally uses the checkpoint's frozen normalization and the same
deterministic test sampler as utils/tokenizer/eval_recon.py, while reporting circular
joint-angle errors in addition to direct EE and FK pose errors.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from robomorpher.utils.data.lazy_data.zarr_dataset import (  # noqa: E402
    ZarrDataset, SequentialBlockBatchSampler, coerce_norm_stats,
)
from robomorpher.model.tokenizer import (  # noqa: E402
    build_tokenizer_from_config, normalize_model_config,
)
from robomorpher.utils.geometry.diff_fk import (  # noqa: E402
    forward_kinematics_6d, rotation_6d_to_matrix,
)
from robomorpher.utils.tokenizer.physical import (  # noqa: E402
    physical_reconstruction, rot_geodesic_deg as geo_deg, error_summary as summary,
)




def per_joint(arr, mask):
    arr = np.asarray(arr)
    mask = np.asarray(mask, dtype=bool)
    out = []
    for j in range(arr.shape[1]):
        out.append(summary(arr[:, j][mask[:, j]]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=".")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--split", default="test", choices=["test", "train", "all"])
    p.add_argument("--sampling", default="first", choices=["first", "even"],
                   help="first uses the evaluator's deterministic prefix; even spreads IDs over the split")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    os.chdir(Path(args.repo).expanduser().resolve())
    cfg = OmegaConf.load(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(ckpt["model_config"])
    vm = normalize_model_config(model_cfg)
    model = build_tokenizer_from_config(model_cfg, state_dict=ckpt["model_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    norm = coerce_norm_stats(ckpt["norm"], normalization_method=str(ckpt["norm"]["normalization_method"]))
    method = str(norm["normalization_method"])
    data_dir = str(cfg.train_data_dir)
    ds = ZarrDataset(
        data_dir, norm, split=args.split, test_split=float(cfg.get("test_split", 0.01)),
        normalization_method=method, urdf_rotation_6d=bool(vm["urdf_rotation_6d"]),
        max_num_joints=int(vm["max_num_joints"]),
    )
    n_total = min(int(args.num_samples), len(ds)) if args.num_samples > 0 else len(ds)
    if n_total <= 0:
        raise ValueError("The requested evaluation split contains no samples")
    if args.sampling == "even" and n_total < len(ds):
        sample_ids = np.linspace(0, len(ds) - 1, n_total, dtype=np.int64)
        eval_ds = torch.utils.data.Subset(ds, sample_ids.tolist())
        loader = torch.utils.data.DataLoader(eval_ds, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0, pin_memory=False)
        sampling_desc = "evenly spaced dataset indices over the deterministic split"
    else:
        loader = torch.utils.data.DataLoader(
            ds, batch_sampler=SequentialBlockBatchSampler(ds, args.batch_size, drop_last=False),
            num_workers=0, pin_memory=False,
        )
        sampling_desc = "SequentialBlockBatchSampler; first requested rows of deterministic split"
    stats = {k: v for k, v in ds.__dict__.items() if k in {
        "x_mean", "x_std", "y_mean", "y_std", "x_min", "x_max", "y_min", "y_max",
        "x_q01", "x_q99", "y_q01", "y_q99",
    }}
    input_with_ee = bool(vm.get("input_with_ee", False))
    q_err, q_abs, ee_pos, ee_rot, fk_pos, fk_rot, masks = [], [], [], [], [], [], []
    seen = 0
    with torch.inference_mode():
        for x, y, mask in loader:
            if seen >= n_total:
                break
            take = min(x.shape[0], n_total - seen)
            x, y, mask = x[:take], y[:take], mask[:take].bool()
            seen += take
            kwargs = {"ee_features": y[..., :9]} if input_with_ee else {}
            ep, jp, _, _, _ = model(x, mask, **kwargs)
            xp, yp, epp, jpp, qgt = physical_reconstruction(x, y, ep, jp, mask, stats, method)
            # Circular revolute-angle distance, in degrees.
            dq = torch.atan2(torch.sin(jpp - qgt), torch.cos(jpp - qgt)).abs().squeeze(-1)
            q_err.append(torch.rad2deg(dq).cpu().numpy())
            q_abs.append(torch.rad2deg((jpp - qgt).abs().squeeze(-1)).cpu().numpy())
            if epp is not None:
                ee_pos.append(torch.linalg.norm(epp[..., :3] - yp[..., :3], dim=-1).mul(1000).cpu().numpy())
                ee_rot.append(geo_deg(epp[..., 3:9], yp[..., 3:9]).cpu().numpy())
            urdf = xp[..., :-1].clone()
            if urdf.shape[-1] == 9:
                flat = urdf.reshape(-1, urdf.shape[-1])
                flat[(~mask).reshape(-1), 3:9] = torch.tensor(
                    [1., 0., 0., 0., 1., 0.], dtype=urdf.dtype
                )
            fk = forward_kinematics_6d(urdf, jpp.squeeze(-1))
            fk_pos.append(torch.linalg.norm(fk[..., :3] - yp[..., :3], dim=-1).mul(1000).cpu().numpy())
            fk_rot.append(geo_deg(fk[..., 3:9], yp[..., 3:9]).cpu().numpy())
            masks.append(mask.cpu().numpy())
    q_err, q_abs, fk_pos, fk_rot, masks = [np.concatenate(v, axis=0) for v in (q_err, q_abs, fk_pos, fk_rot, masks)]
    direct_available = bool(ee_pos)
    ee_pos = np.concatenate(ee_pos, axis=0) if direct_available else None
    ee_rot = np.concatenate(ee_rot, axis=0) if direct_available else None
    def direct_metric(values):
        if values is None:
            return {"available": False, "reason": "angle-only decoder has no direct EE output"}
        return {"aggregate": summary(values[masks]), "per_joint": per_joint(values, masks)}
    result = {
        "checkpoint": str(args.ckpt), "epoch": int(ckpt.get("epoch", -1)),
        "global_step": int(ckpt.get("global_step", -1)), "split": args.split,
        "requested_samples": int(args.num_samples), "seen_samples": int(seen),
        "dataset_rows": int(len(ds)),
        "sampling": sampling_desc,
        "joint_angle_wrap": ds.angle_wrap,
        "primary_joint_angle_metric": "joint_angle_abs_deg" if ds.angle_wrap == "none" else "joint_angle_circular_abs_deg",
        "normalization": method, "input_representation": ("origin_xyz_rot6d_q" if vm["urdf_rotation_6d"] else "origin_xyz_rpy_q"),
        "output_representation": "active_child_xyz_rot6d_q" if direct_available else "angle_only",
        "target_representation": "active_child_xyz_rot6d_q",
        "joint_angle_abs_deg": {"aggregate": summary(q_abs[masks]), "per_joint": per_joint(q_abs, masks)},
        "joint_angle_circular_abs_deg": {"aggregate": summary(q_err[masks]), "per_joint": per_joint(q_err, masks)},
        "ee_direct_position_l2_mm": direct_metric(ee_pos),
        "ee_direct_rotation_geodesic_deg": direct_metric(ee_rot),
        "fk_pred_joint_vs_gt_ee_position_l2_mm": {"aggregate": summary(fk_pos[masks]), "per_joint": per_joint(fk_pos, masks)},
        "fk_pred_joint_vs_gt_ee_rotation_geodesic_deg": {"aggregate": summary(fk_rot[masks]), "per_joint": per_joint(fk_rot, masks)},
        "notes": [
            "EE position is computed in dataset physical units (meters) then multiplied by 1000.",
            "EE rotation is geodesic angle after rot6d-to-matrix conversion.",
            "joint_angle_abs_deg reports direct absolute error; joint_angle_circular_abs_deg also reports the wrapped distance.",
            "Evaluation is model.eval()/inference_mode; VQP EMA/codebook state is not updated.",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

"""Evaluate reconstruction quality of a trained action tokenizer checkpoint.

Physical-space metrics (after de-normalization), on the chosen split:
  - per-joint angle: decoder joint channel vs GT angle (|pred-gt|, rad/deg)
  - per-joint EE: decoder EE channel vs GT EE (pos mm L2 + rot geodesic deg)
  - per-joint diff-FK: FK(urdf, pred joints) vs GT EE at that joint
  - tip-joint FK summary + sanity (FK on GT joints vs GT EE, should be ~0)

Usage (pick a free GPU so training is not disturbed):
    # full dataset (a --num_samples cap selects a deterministic prefix, not a random sample):
    CUDA_VISIBLE_DEVICES=7 python -m robomorpher.utils.tokenizer.eval_recon \
        --config robomorpher/config/tokenizer/robomorpher.yaml --num_samples 2000000
    # held-out test split only:
    CUDA_VISIBLE_DEVICES=7 python -m robomorpher.utils.tokenizer.eval_recon --split test
"""

import argparse
from pathlib import Path
import re
import math
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

import numpy as np
import torch
from omegaconf import OmegaConf

from robomorpher.model.tokenizer import (
    build_tokenizer_from_config,
    normalize_model_config,
)
from robomorpher.utils.geometry.diff_fk import (
    forward_kinematics_6d,
)
from robomorpher.utils.tokenizer.physical import (
    denorm, denorm_features, rot_geodesic_deg, physical_reconstruction, valid_error_summary,
)

from robomorpher.utils.data.lazy_data.zarr_dataset import (
    ZarrDataset, SequentialBlockBatchSampler, coerce_norm_stats, load_norm_stats_cache,
)


def find_latest_ckpt(save_path: str) -> str:
    """Prefer flat run weights; read timestamp runs without lexical epoch sorting."""
    root = Path(save_path).expanduser()
    def candidates(directory):
        return [(int(m.group(1)), f) for f in directory.glob("epoch_*.pt")
                if f.is_file() and (m := re.fullmatch(r"epoch_([0-9]+)\.pt", f.name))]
    direct = candidates(root)
    if direct:
        return str(max(direct, key=lambda item: item[0])[1])
    if root.is_dir():
        reference = sorted((d for d in root.iterdir() if d.is_dir()
                         and re.fullmatch(r"[0-9]{8}_[0-9]{6}", d.name)), reverse=True)
        for directory in reference:
            entries = candidates(directory)
            if entries:
                return str(max(entries, key=lambda item: item[0])[1])
    raise FileNotFoundError(f"No epoch_N.pt checkpoints under {root}; pass --ckpt explicitly")


def resolve_eval_norm(ckpt, cfg, data_dir, num_joints, urdf_rotation_6d):
    """Use frozen normalization, keeping metadata separate from tensor values.

    Only checkpoints without a norm payload may use a historical per-slot cache.
    Never recompute validation statistics or silently replace malformed saved stats.
    """
    raw = ckpt.get("norm")
    if raw is not None:
        if not isinstance(raw, dict) or not raw:
            raise ValueError("Checkpoint has an invalid/empty norm payload; refusing cache fallback")
        method = str(raw.get("normalization_method", cfg.get("normalization_method", "zscore")))
        stats = coerce_norm_stats(raw, normalization_method=method)
        scope = (stats.get("metadata") or {}).get("scope", "per_slot")
        print(f"norm: using checkpoint stats (method={method}, scope={scope})")
        return stats, method
    method = str(cfg.get("normalization_method", "zscore"))
    stem = "norm_stats_rot6d" if urdf_rotation_6d else "norm_stats"
    suffix = "" if method in ("minmax", "zscore") else f"_{method}"
    path = os.path.join(data_dir, f"{stem}_J{num_joints}{suffix}_wrap.npz")
    stats = load_norm_stats_cache(path, normalization_method=method)
    if stats.get("metadata") is not None:
        raise ValueError("A checkpoint without frozen stats cannot infer a new normalization schema")
    print(f"norm: WARNING checkpoint has no norm; reading historical cache {path}. "
          "Exact training-stat provenance cannot be verified; no statistics are recomputed.")
    return stats, method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="robomorpher/config/tokenizer/robomorpher.yaml")
    parser.add_argument("--ckpt", default=None, help="default: highest epoch in save_path (timestamp folders also accepted)")
    parser.add_argument("--split", default="all", choices=["all", "train", "test"],
                        help="all = full dataset (default); test = held-out split")
    parser.add_argument("--num_samples", type=int, default=0, help="0 = whole split")
    parser.add_argument("--data_dir", default=None,
                        help="override cfg.train_data_dir (must contain index.json + norm stats)")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt or find_latest_ckpt(str(cfg.save_path))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"checkpoint: {ckpt_path} (epoch {ckpt.get('epoch')}, step {ckpt.get('global_step')}, "
          f"train_loss {ckpt.get('train_loss')}, test_loss {ckpt.get('test_loss')})")

    # architecture: prefer the config embedded in the checkpoint (frozen at
    # train time) so eval matches the run even if the yaml was edited since
    model_cfg = OmegaConf.create(ckpt["model_config"]) if "model_config" in ckpt else cfg.vqvae_model
    vm = normalize_model_config(model_cfg)
    ee_dim = 9  # Stored targets always have pose9, independently of decoder output.
    joint_dim = int(vm["decoder_joint_dim"])
    input_with_ee = bool(vm.get("input_with_ee", False))
    num_joints = int(vm["max_num_joints"])
    urdf_rotation_6d = bool(vm["urdf_rotation_6d"])

    model = build_tokenizer_from_config(
        model_cfg, state_dict=ckpt["model_state_dict"]
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    data_dir = args.data_dir or str(cfg.train_data_dir)
    # Prefer the ckpt's frozen norm (same stats the model was trained with).
    # Fall back to data_dir npz for older checkpoints that lack "norm".
    norm_stats, method = resolve_eval_norm(ckpt, cfg, data_dir, num_joints, urdf_rotation_6d)

    dataset = ZarrDataset(
        data_dir, norm_stats, split=args.split,
        test_split=float(cfg.get("test_split", 0.01)),
        normalization_method=method,
        urdf_rotation_6d=urdf_rotation_6d,
        max_num_joints=num_joints,
    )
    n_total = len(dataset) if args.num_samples <= 0 else min(args.num_samples, len(dataset))
    if n_total <= 0:
        raise ValueError("The requested evaluation split contains no samples")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=SequentialBlockBatchSampler(dataset, args.batch_size, drop_last=False),
        pin_memory=device.type == "cuda", num_workers=args.num_workers,
    )
    print(f"split={args.split}: {len(dataset)} rows | evaluating on {n_total} samples | method={method} | angle_wrap={dataset.angle_wrap}")

    # move stats to device (minmax uses full range; q99 uses the q01/q99 range)
    stat_keys = {"x_mean", "x_std", "y_mean", "y_std", "x_min", "x_max", "y_min", "y_max",
                 "x_q01", "x_q99", "y_q01", "y_q99"}
    stats = {k: v.to(device) for k, v in dataset.__dict__.items() if k in stat_keys}

    joint_err_all, joint_mask_all = [], []  # (B, J) abs angle err (rad) + mask
    joint_pred_all, joint_gt_all = [], []  # (B, J) physical angle
    ee_pos_err_all, ee_rot_err_all = [], []  # (B, J) decoder EE vs GT EE
    fk_pos_err_all, fk_rot_err_all = [], []  # (B, J) FK(pred joints) vs GT EE
    tip_pos_err_all, tip_rot_err_all = [], []  # tip-joint FK(pred) vs GT EE
    sanity_pos_all, sanity_rot_all, usages_all = [], [], []

    seen = 0
    with torch.no_grad():
        for input_data, output_data, joint_valid_mask in loader:
            if seen >= n_total:
                break
            input_data = input_data.to(device, non_blocking=True)
            output_data = output_data.to(device, non_blocking=True)
            joint_valid_mask = joint_valid_mask.to(device, non_blocking=True).bool()
            b = input_data.shape[0]
            take = min(b, n_total - seen)
            input_data = input_data[:take]
            output_data = output_data[:take]
            joint_valid_mask = joint_valid_mask[:take]
            b = take
            seen += b

            encoder_kwargs = ({"ee_features": output_data[..., :ee_dim]}
                              if input_with_ee else {})
            ee_pred, joint_pred, _, _, usages = model(
                input_data, joint_valid_mask, **encoder_kwargs,
            )
            usages_all.append(usages.float().cpu())

            x_phys, y_phys, ee_pred_phys, joint_pred_phys, joint_gt_phys = physical_reconstruction(
                input_data, output_data, ee_pred, joint_pred, joint_valid_mask, stats, method
            )

            # ---- per-joint absolute angle error (rad)
            joint_gt_phys = y_phys[..., ee_dim:ee_dim + joint_dim]
            abs_err = (joint_pred_phys - joint_gt_phys).abs().squeeze(-1)  # (B, J)
            joint_err_all.append(abs_err.cpu())
            joint_pred_all.append(joint_pred_phys.squeeze(-1).cpu())
            joint_gt_all.append(joint_gt_phys.squeeze(-1).cpu())
            joint_mask_all.append(joint_valid_mask.cpu())

            # ---- per-joint decoder EE vs GT EE
            ee_gt_all = y_phys[..., :ee_dim]
            if ee_pred_phys is not None:
                ee_pos_err_all.append(
                    torch.linalg.norm(ee_pred_phys[..., :3] - ee_gt_all[..., :3], dim=-1).cpu()
                )
                ee_rot_err_all.append(
                    rot_geodesic_deg(ee_pred_phys[..., 3:9], ee_gt_all[..., 3:9]).cpu()
                )

            # ---- FK on predicted / GT joint angles -> pose at every joint
            urdf_features = x_phys[..., :-1]  # (B, J, 9|rot6d or 6|rpy), angle channel dropped
            if urdf_features.shape[-1] == 9:
                # padding joints denormalize to all-zero rot6d (invalid) -> identity.
                # NOTE: `urdf[~mask, 3:9] = v` is invalid torch semantics; use a
                # flattened 1D mask instead.
                urdf_features = urdf_features.clone()
                flat = urdf_features.reshape(-1, urdf_features.shape[-1])
                flat[(~joint_valid_mask).reshape(-1), 3:9] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=urdf_features.dtype
                )
            fk_pred = forward_kinematics_6d(urdf_features, joint_pred_phys.squeeze(-1))
            fk_gt = forward_kinematics_6d(urdf_features, joint_gt_phys.squeeze(-1))

            fk_pos_err_all.append(
                torch.linalg.norm(fk_pred[..., :3] - ee_gt_all[..., :3], dim=-1).cpu()
            )
            fk_rot_err_all.append(
                rot_geodesic_deg(fk_pred[..., 3:9], ee_gt_all[..., 3:9]).cpu()
            )

            last_valid = num_joints - 1 - joint_valid_mask.long().fliplr().argmax(dim=1)
            bidx = torch.arange(b, device=device)
            ee_gt_tip = y_phys[bidx, last_valid, :ee_dim]  # (B, 9) xyz+rot6d
            pose_pred = fk_pred[bidx, last_valid]          # (B, 9)
            pose_gt = fk_gt[bidx, last_valid]              # (B, 9) sanity

            has_joint = joint_valid_mask.any(dim=1)
            tip_pos_err_all.append((pose_pred[has_joint, :3] - ee_gt_tip[has_joint, :3]).abs().cpu())
            tip_rot_err_all.append(rot_geodesic_deg(pose_pred[has_joint, 3:9], ee_gt_tip[has_joint, 3:9]).cpu())
            sanity_pos_all.append((pose_gt[has_joint, :3] - ee_gt_tip[has_joint, :3]).abs().cpu())
            sanity_rot_all.append(rot_geodesic_deg(pose_gt[has_joint, 3:9], ee_gt_tip[has_joint, 3:9]).cpu())

    # ---------------- report
    jerr = torch.cat(joint_err_all).numpy()       # (N, J) rad
    jpred = torch.cat(joint_pred_all).numpy()     # (N, J) rad
    jgt = torch.cat(joint_gt_all).numpy()         # (N, J) rad
    jmask = torch.cat(joint_mask_all).numpy().astype(bool)  # (N, J)
    fk_pos = torch.cat(fk_pos_err_all).numpy() * 1000.0     # (N, J) mm
    fk_rot = torch.cat(fk_rot_err_all).numpy()              # (N, J) deg
    tip_pos = torch.cat(tip_pos_err_all).numpy()            # (N, 3) meters
    tip_rot = torch.cat(tip_rot_err_all).numpy()            # (N,) deg
    spos = torch.cat(sanity_pos_all).numpy()
    srot = torch.cat(sanity_rot_all).numpy()
    tip_pos_norm = np.linalg.norm(tip_pos, axis=-1) * 1000.0  # mm
    spos_norm = np.linalg.norm(spos, axis=-1) * 1000.0

    print(f"\n===== reconstruction quality ({seen} samples) =====")
    print(f"codebook usage: {torch.stack(usages_all).mean().item():.2f}%")  # already in percent

    print("\n[1] per-joint angle: decoder joint vs GT (physical rad)")
    print(f"  {'joint':>5} {'mean_pred':>10} {'mean_gt':>10} {'MAE(rad)':>10} "
          f"{'MAE(deg)':>10} {'p90(deg)':>10} {'n':>8}")
    mae_all, p90_all = [], []
    for j in range(num_joints):
        m = jmask[:, j]
        if not m.any():
            print(f"  {j:>5} {'n/a':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10} {0:>8}")
            continue
        e = jerr[:, j][m]
        mae_all.append(e.mean())
        p90_all.append(np.percentile(e, 90))
        print(f"  {j:>5} {jpred[:, j][m].mean():>10.5f} {jgt[:, j][m].mean():>10.5f} "
              f"{mae_all[-1]:>10.5f} {math.degrees(mae_all[-1]):>10.3f} "
              f"{math.degrees(p90_all[-1]):>10.3f} {m.sum():>8}")
    if mae_all:
        pooled_angle = valid_error_summary(jerr, jmask)
        print(f"  {'all':>5} {'':>10} {'':>10} {pooled_angle['mean']:>10.5f} "
              f"{math.degrees(pooled_angle['mean']):>10.3f} "
              f"{math.degrees(pooled_angle['p90']):>10.3f}")

    if ee_pos_err_all:
        ee_pos = torch.cat(ee_pos_err_all).numpy() * 1000.0     # (N, J) mm
        ee_rot = torch.cat(ee_rot_err_all).numpy()              # (N, J) deg
        print("\n[2] per-joint EE: decoder EE vs GT EE (physical)")
        print(f"  {'joint':>5} {'pos_mean':>10} {'pos_p90':>10} {'rot_mean':>10} "
              f"{'rot_p90':>10} {'n':>8}")
        print(f"  {'':>5} {'(mm)':>10} {'(mm)':>10} {'(deg)':>10} {'(deg)':>10}")
        ee_pos_means, ee_rot_means = [], []
        for j in range(num_joints):
            m = jmask[:, j]
            if not m.any():
                print(f"  {j:>5} {'n/a':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10} {0:>8}")
                continue
            p, r = ee_pos[:, j][m], ee_rot[:, j][m]
            ee_pos_means.append(p.mean())
            ee_rot_means.append(r.mean())
            print(f"  {j:>5} {p.mean():>10.2f} {np.percentile(p, 90):>10.2f} "
                  f"{r.mean():>10.3f} {np.percentile(r, 90):>10.3f} {m.sum():>8}")
        if ee_pos_means:
            print(f"  {'all':>5} {valid_error_summary(ee_pos, jmask)['mean']:>10.2f} {'':>10} "
                  f"{valid_error_summary(ee_rot, jmask)['mean']:>10.3f}")
    else:
        print("\n[2] per-joint EE: skipped (no decoder-EE samples collected)")

    print("\n[3] per-joint diff-FK: FK(urdf, pred joints) vs GT EE (physical)")
    print(f"  {'joint':>5} {'pos_mean':>10} {'pos_p90':>10} {'rot_mean':>10} "
          f"{'rot_p90':>10} {'n':>8}")
    print(f"  {'':>5} {'(mm)':>10} {'(mm)':>10} {'(deg)':>10} {'(deg)':>10}")
    fk_pos_means, fk_rot_means = [], []
    for j in range(num_joints):
        m = jmask[:, j]
        if not m.any():
            print(f"  {j:>5} {'n/a':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10} {0:>8}")
            continue
        p, r = fk_pos[:, j][m], fk_rot[:, j][m]
        fk_pos_means.append(p.mean())
        fk_rot_means.append(r.mean())
        print(f"  {j:>5} {p.mean():>10.2f} {np.percentile(p, 90):>10.2f} "
              f"{r.mean():>10.3f} {np.percentile(r, 90):>10.3f} {m.sum():>8}")
    if fk_pos_means:
        print(f"  {'all':>5} {valid_error_summary(fk_pos, jmask)['mean']:>10.2f} {'':>10} "
              f"{valid_error_summary(fk_rot, jmask)['mean']:>10.3f}")

    if not tip_pos_norm.size:
        print("\n[tip] skipped: no sample has a valid joint")
        return

    print("\n[tip] EE pose after diff-FK on predicted joints (last valid joint vs GT ee):")
    print(f"  position: mean {tip_pos_norm.mean():.2f} mm | p50 {np.percentile(tip_pos_norm, 50):.2f} mm "
          f"| p90 {np.percentile(tip_pos_norm, 90):.2f} mm | max {tip_pos_norm.max():.2f} mm")
    print(f"  per-axis abs: x {tip_pos[:, 0].mean()*1000:.2f} mm, y {tip_pos[:, 1].mean()*1000:.2f} mm, "
          f"z {tip_pos[:, 2].mean()*1000:.2f} mm")
    print(f"  rotation: mean {tip_rot.mean():.3f} deg | p50 {np.percentile(tip_rot, 50):.3f} deg "
          f"| p90 {np.percentile(tip_rot, 90):.3f} deg")

    print("\nsanity (FK on GT joints vs GT ee channel, should be ~0):")
    print(f"  position: mean {spos_norm.mean():.3f} mm | p90 {np.percentile(spos_norm, 90):.3f} mm")
    print(f"  rotation: mean {srot.mean():.4f} deg | p90 {np.percentile(srot, 90):.4f} deg")


if __name__ == "__main__":
    main()

"""
Compute per-subject uncertainty bundle features from 5 Anatomix seed outputs.

Features computed per subject:
  def_variance               — mean magnitude of per-voxel variance in the STN
                               transform field across seeds (registration instability)
  volume_variance            — variance of coronary mask voxel count across seeds
                               (topology/extent instability)
  pairwise_dice_disagreement — mean pairwise (1 – Dice) across all 10 seed pairs
                               (segmentation inconsistency)

Inputs:
  Transform fields: uncertainty_root/seed_*/sample_X_transform.nii.gz
  Segmentation:     uncertainty_root/seed_*/sample_X_image_prime_argmax.nii.gz

Output: {out_dir}/metrics/uncertainty_bundle_features.csv
Columns: sample_id, def_variance, volume_variance, pairwise_dice_disagreement

Does NOT overwrite any existing uncertainty CSVs (entropy or defvar).
Only writes uncertainty_bundle_features.csv (a new unique filename).

Usage (test split, fold 1):
  python coronary-compute-unc-bundle.py \
      --uncertainty_root output/asoca-coronary/fold_1/full-stn/uncertainty \
      --out_dir          output/asoca-coronary/fold_1/full-stn/uncertainty_analysis

Usage (train split, fold 1):
  python coronary-compute-unc-bundle.py \
      --uncertainty_root output/asoca-coronary/fold_1/full-stn-train-unc/uncertainty \
      --out_dir          output/asoca-coronary/fold_1/full-stn-train-unc/uncertainty_analysis
"""

import argparse
import csv
import re
from pathlib import Path

import nibabel as nib
import numpy as np


# Constants


SEEDS = [42, 123, 456, 789, 999]
CORONARY_LABEL = 1  # label index for coronary artery in argmax labelmaps


# I/O helpers


def load_transform(path: Path) -> np.ndarray:
    """Load a NIfTI transform field saved by SimpleITK (isVector=True).

    SimpleITK saves a (D, H, W, 3) vector field as a NIfTI that nibabel reads
    back as either (D, H, W, 3) [4D] or (D, H, W, 1, 3) [5D with time singleton].
    Both shapes are handled here; output is always (D, H, W, 3) float32.
    """
    data = nib.load(path).get_fdata()
    if data.ndim == 4 and data.shape[-1] == 3:
        return data.astype(np.float32)
    if data.ndim == 5 and data.shape[3] == 1 and data.shape[4] == 3:
        return data[:, :, :, 0, :].astype(np.float32)
    raise ValueError(
        f"Unexpected transform shape {data.shape} at {path}. "
        f"Expected (D,H,W,3) or (D,H,W,1,3)."
    )



# Core computation


def compute_deformation_variance_mean(
    uncertainty_root: Path,
    sample_id: int,
    seeds: list,
) -> float:
    """Load 5 transform fields, compute mean of voxelwise variance magnitude.

    Algorithm:
        fields.shape = (N, D, H, W, 3)
        var_field     = np.var(fields, axis=0)   # (D, H, W, 3)
        mag           = np.linalg.norm(var_field, axis=-1)  # (D, H, W)
        result        = mag.mean()

    Missing files: replaced by zero fields (safe default; reduces sensitivity
    but does not crash). Returns 0.0 if all files are missing or result is NaN.
    """
    fields = []
    ref_shape = None

    for seed in seeds:
        path = (
            uncertainty_root / f"seed_{seed}"
            / f"sample_{sample_id}_transform.nii.gz"
        )
        if not path.exists():
            print(f"  [WARN] Missing: {path} — substituting zeros")
            fields.append(None)
        else:
            arr = load_transform(path)
            if ref_shape is None:
                ref_shape = arr.shape
            fields.append(arr)

    valid = [f for f in fields if f is not None]
    if len(valid) == 0:
        print(f"  [WARN] No transform fields found for sample {sample_id}; returning 0")
        return 0.0

    if ref_shape is None:
        ref_shape = valid[0].shape

    filled = [
        f if f is not None else np.zeros(ref_shape, dtype=np.float32)
        for f in fields
    ]

    stack = np.stack(filled, axis=0)         # (N, D, H, W, 3)
    var_field = np.var(stack, axis=0)        # (D, H, W, 3)
    mag = np.linalg.norm(var_field, axis=-1) # (D, H, W)

    val = float(mag.mean())
    if not np.isfinite(val):
        print(f"  [WARN] Non-finite result for sample {sample_id}; returning 0")
        return 0.0
    return val



# insert new bundle features: volume variance and pairwise Dice disagreement

def load_argmax_mask(path: Path) -> np.ndarray:
    """Load a NIfTI image_prime_argmax and return the binary coronary artery mask.

    image_prime_argmax is the argmax of the Anatomix segmentation of the subject
    image warped to atlas space. Values: 0 = background, 1 = coronary artery.
    warped_atlas_labelmap_argmax is NOT used here — it is all zeros because the
    atlas coronary probability is too diffuse to ever win argmax.
    """
    data = nib.load(path).get_fdata()
    return (data == CORONARY_LABEL).astype(np.float32)


def compute_volume_variance(
    uncertainty_root: Path,
    sample_id: int,
    seeds: list,
) -> float:
    """Variance of coronary mask voxel count across seeds.

    Algorithm:
        volumes = [mask.sum() for each seed]
        result  = np.var(volumes)

    Returns 0.0 if fewer than 2 masks are available.
    """
    volumes = []
    for seed in seeds:
        path = (
            uncertainty_root / f"seed_{seed}"
            / f"sample_{sample_id}_image_prime_argmax.nii.gz"
        )
        if not path.exists():
            print(f"  [WARN] Missing: {path} — skipping seed {seed}")
            continue
        mask = load_argmax_mask(path)
        volumes.append(float(mask.sum()))

    if len(volumes) < 2:
        print(f"  [WARN] Fewer than 2 argmax masks for sample {sample_id}; returning 0")
        return 0.0

    val = float(np.var(volumes))
    if not np.isfinite(val):
        print(f"  [WARN] Non-finite volume_variance for sample {sample_id}; returning 0")
        return 0.0
    return val


def compute_pairwise_dice_disagreement(
    uncertainty_root: Path,
    sample_id: int,
    seeds: list,
) -> float:
    """Mean pairwise Dice disagreement (1 – Dice) across all seed pairs.

    Algorithm:
        For N seeds -> N*(N-1)/2 pairs:
            dice = 2 * |A ∩ B| / (|A| + |B|)
            disagreement = 1 - dice
        result = mean(disagreements)

    Empty intersection of both masks -> dice = 1 (perfect agreement on emptiness).
    Returns 0.0 if fewer than 2 masks are available.
    """
    masks = []
    for seed in seeds:
        path = (
            uncertainty_root / f"seed_{seed}"
            / f"sample_{sample_id}_image_prime_argmax.nii.gz"
        )
        if not path.exists():
            print(f"  [WARN] Missing: {path} — skipping seed {seed}")
            continue
        masks.append(load_argmax_mask(path))

    if len(masks) < 2:
        print(f"  [WARN] Fewer than 2 argmax masks for sample {sample_id}; returning 0")
        return 0.0

    disagreements = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            denom = masks[i].sum() + masks[j].sum()
            dice = float(2.0 * (masks[i] * masks[j]).sum() / denom) if denom > 0 else 1.0
            disagreements.append(1.0 - dice)

    val = float(np.mean(disagreements))
    if not np.isfinite(val):
        print(f"  [WARN] Non-finite pairwise_dice_disagreement for sample {sample_id}; returning 0")
        return 0.0
    return val



# Discovery helpers


def discover_sample_ids(uncertainty_root: Path, seeds: list) -> list:
    """Infer sample indices by scanning the first seed folder for transform files."""
    first_seed_dir = uncertainty_root / f"seed_{seeds[0]}"
    if not first_seed_dir.exists():
        raise FileNotFoundError(
            f"First seed directory not found: {first_seed_dir}"
        )
    pattern = re.compile(r"sample_(\d+)_transform\.nii\.gz")
    ids = sorted(
        int(m.group(1))
        for f in first_seed_dir.iterdir()
        if (m := pattern.match(f.name))
    )
    if not ids:
        raise RuntimeError(
            f"No sample_*_transform.nii.gz files found in {first_seed_dir}"
        )
    return ids



# Main


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute uncertainty bundle features from multi-seed STN outputs. "
            "Writes uncertainty_bundle_features.csv — does NOT overwrite other CSVs."
        )
    )
    parser.add_argument(
        "--uncertainty_root",
        default="output/asoca-coronary/fold_1/full-stn/uncertainty",
        help=(
            "Root folder containing seed_* subdirectories, each with "
            "sample_X_transform.nii.gz files."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="output/asoca-coronary/fold_1/full-stn/uncertainty_analysis",
        help=(
            "Root output directory. The CSV is written to {out_dir}/metrics/. "
            "Existing files in that directory are NOT touched."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seed values to aggregate (default: 42 123 456 789 999).",
    )
    args = parser.parse_args()

    uncertainty_root = Path(args.uncertainty_root)
    out_root = Path(args.out_dir)
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Uncertainty root : {uncertainty_root}")
    print(f"Output metrics   : {metrics_dir}")
    print(f"Seeds            : {args.seeds}")

    sample_ids = discover_sample_ids(uncertainty_root, args.seeds)
    print(f"Found sample IDs : {sample_ids}  ({len(sample_ids)} subjects)")

    rows = []
    for sample_id in sample_ids:
        print(f"\n--- Sample {sample_id} ---")
        defvar  = compute_deformation_variance_mean(uncertainty_root, sample_id, args.seeds)
        vol_var = compute_volume_variance(uncertainty_root, sample_id, args.seeds)
        pw_dice = compute_pairwise_dice_disagreement(uncertainty_root, sample_id, args.seeds)
        rows.append({
            "sample_id":                    sample_id,
            "def_variance":                 defvar,
            "volume_variance":              vol_var,
            "pairwise_dice_disagreement":   pw_dice,
        })
        print(f"  def_variance               = {defvar:.8f}")
        print(f"  volume_variance            = {vol_var:.4f}")
        print(f"  pairwise_dice_disagreement = {pw_dice:.6f}")

    out_csv = metrics_dir / "uncertainty_bundle_features.csv"
    fieldnames = ["sample_id", "def_variance", "volume_variance", "pairwise_dice_disagreement"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_csv}  ({len(rows)} rows)")

    print(f"\nSummary statistics across {len(rows)} subjects:")
    for col in ["def_variance", "volume_variance", "pairwise_dice_disagreement"]:
        vals = [r[col] for r in rows]
        mean_v = np.mean(vals)
        print(f"\n  {col}:")
        print(f"    mean = {mean_v:.6f}")
        print(f"    std  = {np.std(vals):.6f}")
        print(f"    min  = {np.min(vals):.6f}")
        print(f"    max  = {np.max(vals):.6f}")
        if mean_v > 0:
            print(f"    CV   = {np.std(vals)/mean_v:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Compute per-subject deformation_variance_mean from 5 Anatomix seed transform fields.

For each subject, loads sample_X_transform.nii.gz from each seed directory under
uncertainty_root/seed_*/. Stacks the 5 fields, computes voxelwise variance across
seeds, then takes the mean magnitude of that variance field.

This measures how unstable the inferred anatomy/registration is when the input
segmentation changes slightly between seeds — a much more meaningful signal than
voxel entropy, because it measures deformation instability rather than segmentation
fuzziness.

Output: {out_dir}/metrics/deformation_variance_summary.csv
Columns: sample_id, deformation_variance_mean

Does NOT overwrite any existing entropy-based uncertainty_analysis CSVs.
Only writes deformation_variance_summary.csv (a new unique filename).

Usage (test split, fold 1):
  python coronary-compute-defvar.py \
      --uncertainty_root output/asoca-coronary/fold_1/full-stn/uncertainty \
      --out_dir          output/asoca-coronary/fold_1/full-stn/uncertainty_analysis

Usage (train split, fold 1):
  python coronary-compute-defvar.py \
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
            "Compute deformation_variance_mean from multi-seed STN transform fields. "
            "Writes deformation_variance_summary.csv — does NOT overwrite entropy CSVs."
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
        defvar = compute_deformation_variance_mean(
            uncertainty_root, sample_id, args.seeds
        )
        rows.append({
            "sample_id":                sample_id,
            "deformation_variance_mean": defvar,
        })
        print(f"  deformation_variance_mean = {defvar:.8f}")

    out_csv = metrics_dir / "deformation_variance_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "deformation_variance_mean"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_csv}  ({len(rows)} rows)")

    vals = [r["deformation_variance_mean"] for r in rows]
    print(f"\nSummary statistics across {len(vals)} subjects:")
    print(f"  mean  = {np.mean(vals):.6f}")
    print(f"  std   = {np.std(vals):.6f}")
    print(f"  min   = {np.min(vals):.6f}")
    print(f"  max   = {np.max(vals):.6f}")
    print(f"  CV    = {np.std(vals)/np.mean(vals):.4f}" if np.mean(vals) > 0 else "  CV = N/A")
    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Compute Normalised Cross Correlation (NCC) between uncertainty maps and
voxel-wise error maps for the MM-WHS test set.

NCC is the Pearson correlation coefficient computed over all voxels:
  NCC(A, B) = sum[(A - mean_A)(B - mean_B)] / sqrt[sum(A-mean_A)^2 * sum(B-mean_B)^2]

Pairings computed per sample (6 test images):
  1. Multiclass error  vs  multiclass entropy (multi-seed ensemble)
  2. Per-class error k vs  per-class entropy k (multi-seed ensemble),  for k = 1..7
  3. Multiclass error  vs  multiclass KL divergence (multi-seed ensemble)
  4. Per-class error k vs  per-class KL divergence k (multi-seed ensemble),  for k = 1..7
  5. Multiclass error  vs  multiclass aleatoric uncertainty
  6. Per-class error k vs  per-class aleatoric uncertainty k,  for k = 1..7

Results: per-sample NCC values, plus mean and variance across the 6 samples.
Saved to: uncertainty_analysis/ncc_results/ncc_results.csv
"""

import csv
from pathlib import Path

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path("output/mm-whs/full-stn")

ERROR_DIR        = BASE / "uncertainty_analysis" / "error_maps"
ENTROPY_MULTI    = BASE / "uncertainty_analysis" / "entropy_maps"
ENTROPY_PC       = BASE / "uncertainty_analysis" / "per_class_entropy_maps"
KL_MULTI         = BASE / "uncertainty_analysis_kl" / "kl_divergence_maps"
KL_PC            = BASE / "uncertainty_analysis_kl" / "per_class_kl_maps"
ALEATORIC_MULTI  = BASE / "uncertainty_analysis_aleatoric" / "kl_and_aleatoric_divergence_maps"
ALEATORIC_PC     = BASE / "uncertainty_analysis_aleatoric" / "per_class_aleatoric_maps"
OUT_DIR          = BASE / "uncertainty_analysis" / "ncc_results"

N_SAMPLES        = 6
FOREGROUND_CLASSES = list(range(1, 8))   # 1..7

CLASS_NAMES = {
    1: "Myocardium",
    2: "Left_Atrium",
    3: "Left_Ventricle",
    4: "Right_Atrium",
    5: "Right_Ventricle",
    6: "Aorta",
    7: "Pulmonary_Artery",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32).ravel()


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient over all voxels."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # (label, {sample_index: ncc_value})
    metrics: list[tuple[str, dict[int, float]]] = []

    # ---- 1. Multiclass error vs multiclass entropy (multi-seed) -----------

    label = "Multiclass error | Entropy (multi-seed)"
    per_sample: dict[int, float] = {}
    for s in range(N_SAMPLES):
        err_path = ERROR_DIR / f"sample_{s}_error_multiclass.nii.gz"
        unc_path = ENTROPY_MULTI / f"sample_{s}_uncertainty_entropy.nii.gz"
        if not err_path.exists() or not unc_path.exists():
            print(f"  [missing] sample {s}: {err_path} or {unc_path}")
            per_sample[s] = float("nan")
            continue
        per_sample[s] = ncc(load(err_path), load(unc_path))
    metrics.append((label, per_sample))

    # ---- 2. Per-class error vs per-class entropy (single-seed) -----------

    for k in FOREGROUND_CLASSES:
        label = f"Class {k} {CLASS_NAMES[k]} error | Entropy (multi-seed)"
        per_sample = {}
        for s in range(N_SAMPLES):
            err_path = ERROR_DIR / f"sample_{s}_error_class_{k}_{CLASS_NAMES[k]}.nii.gz"
            unc_path = ENTROPY_PC / f"sample_{s}_class_{k}_entropy.nii.gz"
            if not err_path.exists() or not unc_path.exists():
                print(f"  [missing] sample {s}: {err_path} or {unc_path}")
                per_sample[s] = float("nan")
                continue
            per_sample[s] = ncc(load(err_path), load(unc_path))
        metrics.append((label, per_sample))

    # ---- 3. Multiclass error vs multiclass KL divergence (multi-seed) -----

    label = "Multiclass error | KL divergence (multi-seed)"
    per_sample = {}
    for s in range(N_SAMPLES):
        err_path = ERROR_DIR / f"sample_{s}_error_multiclass.nii.gz"
        unc_path = KL_MULTI   / f"sample_{s}_avg_kl_divergence.nii.gz"
        if not err_path.exists() or not unc_path.exists():
            print(f"  [missing] sample {s}: {err_path} or {unc_path}")
            per_sample[s] = float("nan")
            continue
        per_sample[s] = ncc(load(err_path), load(unc_path))
    metrics.append((label, per_sample))

    # ---- 4. Per-class error vs per-class KL divergence (multi-seed) -------

    for k in FOREGROUND_CLASSES:
        label = f"Class {k} {CLASS_NAMES[k]} error | KL divergence (multi-seed)"
        per_sample = {}
        for s in range(N_SAMPLES):
            err_path = ERROR_DIR / f"sample_{s}_error_class_{k}_{CLASS_NAMES[k]}.nii.gz"
            unc_path = KL_PC     / f"sample_{s}_class_{k}_kl_divergence.nii.gz"
            if not err_path.exists() or not unc_path.exists():
                print(f"  [missing] sample {s}: {err_path} or {unc_path}")
                per_sample[s] = float("nan")
                continue
            per_sample[s] = ncc(load(err_path), load(unc_path))
        metrics.append((label, per_sample))

    # ---- 5. Multiclass error vs multiclass aleatoric uncertainty ----------

    label = "Multiclass error | Aleatoric uncertainty"
    per_sample = {}
    for s in range(N_SAMPLES):
        err_path = ERROR_DIR      / f"sample_{s}_error_multiclass.nii.gz"
        unc_path = ALEATORIC_MULTI / f"sample_{s}_aleatoric_map.nii.gz"
        if not err_path.exists() or not unc_path.exists():
            print(f"  [missing] sample {s}: {err_path} or {unc_path}")
            per_sample[s] = float("nan")
            continue
        per_sample[s] = ncc(load(err_path), load(unc_path))
    metrics.append((label, per_sample))

    # ---- 6. Per-class error vs per-class aleatoric uncertainty ------------

    for k in FOREGROUND_CLASSES:
        label = f"Class {k} {CLASS_NAMES[k]} error | Aleatoric uncertainty"
        per_sample = {}
        for s in range(N_SAMPLES):
            err_path = ERROR_DIR    / f"sample_{s}_error_class_{k}_{CLASS_NAMES[k]}.nii.gz"
            unc_path = ALEATORIC_PC / f"sample_{s}_class_{k}_aleatoric.nii.gz"
            if not err_path.exists() or not unc_path.exists():
                print(f"  [missing] sample {s}: {err_path} or {unc_path}")
                per_sample[s] = float("nan")
                continue
            per_sample[s] = ncc(load(err_path), load(unc_path))
        metrics.append((label, per_sample))

    # ---- Aggregate statistics ---------------------------------------------

    rows = []
    for label, per_sample in metrics:
        vals = np.array([per_sample[s] for s in range(N_SAMPLES)], dtype=np.float64)
        valid = vals[~np.isnan(vals)]
        mean = float(np.mean(valid)) if len(valid) else float("nan")
        var  = float(np.var(valid, ddof=0)) if len(valid) else float("nan")

        row = {"metric": label}
        for s in range(N_SAMPLES):
            v = per_sample[s]
            row[f"sample_{s}"] = f"{v:.6f}" if not np.isnan(v) else "nan"
        row["mean"]     = f"{mean:.6f}"
        row["variance"] = f"{var:.6f}"
        rows.append(row)

    # ---- Save CSV ---------------------------------------------------------

    csv_path = OUT_DIR / "ncc_results.csv"
    fieldnames = ["metric"] + [f"sample_{s}" for s in range(N_SAMPLES)] + ["mean", "variance"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}\n")

    # ---- Print table ------------------------------------------------------

    col_w = 55
    num_w = 10

    header = f"{'Metric':<{col_w}}" + "".join(f"{'s'+str(s):>{num_w}}" for s in range(N_SAMPLES))
    header += f"{'Mean':>{num_w}}{'Variance':>{num_w}}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for row in rows:
        line = f"{row['metric']:<{col_w}}"
        for s in range(N_SAMPLES):
            line += f"{row[f'sample_{s}']:>{num_w}}"
        line += f"{row['mean']:>{num_w}}{row['variance']:>{num_w}}"
        print(line)

    print(sep)
    print(f"\nAll values are Pearson NCC in [-1, 1].  Rows: {len(rows)}.  Samples: {N_SAMPLES}.")


if __name__ == "__main__":
    main()

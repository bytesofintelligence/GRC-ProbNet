"""
Post-processing script: compute voxel-wise uncertainty across multiple Anatomix seeds.

It is the bridge between "test_uncertainty()" which generates warped atlas outputs
for each seed, and the Geo-Radio-Classification script which consumes the uncertainty metrics.

Each seed folder contains warped atlas outputs from a single STN forward pass with
a different Anatomix pseudo-label seed. Uncertainty is measured as the spread of
predicted soft probability maps across those seeds.

Soft labelmap NIfTI files have shape (D, H, W, 1, C) where axis-3 is a saved batch
dimension. Per-structure uncertainty uses the average KL divergence from the mean
(Jensen-Shannon Divergence / JSD) with soft-mask thresholding to localise the JSD
signal to each structure's region. Threshold sweep over multiple probability thresholds
selects the one that best discriminates per-structure JSD across samples.
"""

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np

# Constants

SEEDS = [42, 123, 456, 789, 999]

CLASS_NAMES = {
    0: "Background",
    1: "Myocardium",
    2: "Left Atrium",
    3: "Left Ventricle",
    4: "Right Atrium",
    5: "Right Ventricle",
    6: "Aorta",
    7: "Pulmonary Artery",
}

# Probability thresholds evaluated during the soft-mask sweep
THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.1]


# I/O helpers

def load_soft_labelmap(path: Path) -> np.ndarray:
    """Load soft probability map, squeeze batch dimension, return (D, H, W, C).
    """
    data = nib.load(path).get_fdata()
    assert data.ndim == 5 and data.shape[3] == 1, (
        f"Expected shape (D,H,W,1,C), got {data.shape} for {path}"
    )
    return data[:, :, :, 0, :]  # (D, H, W, C)


def load_argmax(path: Path) -> np.ndarray:
    """Load argmax segmentation map, shape (D, H, W)."""
    return nib.load(path).get_fdata().astype(np.int32)


def save_nifti(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    """Save array using the affine and header from ref_img."""
    out = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(out, str(out_path))

# Uncertainty computations

def compute_mean_probs(stack: np.ndarray) -> np.ndarray:
    """Mean over seed axis. Input (N, D, H, W, C), output (D, H, W, C)."""
    return stack.mean(axis=0)


def compute_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Multiclass predictive entropy from probability map.
    H = -sum_c [ p_c * log(p_c + eps) ]

    Input:  (D, H, W, C)
    Output: (D, H, W)
    """
    return -(probs * np.log(probs + eps)).sum(axis=-1)


def compute_binary_class_entropy(prob_map: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Binary entropy for a single-class probability map.

    Treats each voxel as a Bernoulli variable with p = predicted probability for this class:
        H_c = -p * log(p) - (1-p) * log(1-p)
    Unlike multiclass entropy over argmax voxels, this captures
    uncertainty at class boundaries and near-miss voxels with exclude completely

    Input:  (D, H, W)  values in [0, 1]
    Output: (D, H, W)  values in [0, log(2)]
    """
    p = np.clip(prob_map, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """KL divergence KL(p || q) per voxel.

    KL(p || q) = sum_c [ p_c * log(p_c / q_c) ]

    Inputs:  (D, H, W, C) each
    Output:  (D, H, W)
    """
    return (p * np.log((p + eps) / (q + eps))).sum(axis=-1)


def compute_kl_divergence_per_class(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-class KL divergence contribution KL_c(p || q) per voxel, without summing over classes.

    KL_c = p_c * log(p_c / q_c)

    Inputs:  (D, H, W, C) each
    Output:  (D, H, W, C)
    """
    return p * np.log((p + eps) / (q + eps))


def compute_disagreement(argmax_stack: np.ndarray) -> np.ndarray:
    """Number of unique predicted labels per voxel across seeds.

    Input:  (N, D, H, W)  int32
    Output: (D, H, W)     uint8  range [1, N]
    """
    N, D, H, W = argmax_stack.shape
    flat = argmax_stack.reshape(N, -1)  # (N, D*H*W)
    disagreement = np.array(
        [len(np.unique(flat[:, v])) for v in range(flat.shape[1])],
        dtype=np.uint8,
    ).reshape(D, H, W)
    return disagreement


# Threshold sweep for per-structure JSD (average KL from mean)

def compute_threshold_sweep(
    mean_probs: np.ndarray,
    avg_kl: np.ndarray,
    sample_id: int,
    thresholds: list,
) -> list:
    """Masked mean JSD for each foreground class at every threshold τ.

    For class c and threshold τ, restrict the avg_kl average to voxels where
    mean_probs[..., c] > τ. This localises the global JSD map to each structure's
    region and excludes near-zero background voxels that would dilute the estimate.
    Background (class 0) is skipped.

    avg_kl is the Jensen-Shannon Divergence map: average over seeds of
    KL(seed_i || mean_probs), shape (D, H, W). One scalar per voxel measuring
    total multiclass seed disagreement at that location.

    Args:
        mean_probs:  (D, H, W, C) mean probability map across seeds
        avg_kl:      (D, H, W)   average KL divergence from mean (JSD map)
        sample_id:   integer sample index for labelling rows
        thresholds:  list of float probability thresholds

    Returns a list of dicts with keys:
            sample_id, class_id, class_name, threshold, mean_kl, voxel_count
    """
    rows = []
    C = mean_probs.shape[-1]
    for class_id in range(1, C):  # skip background
        p = mean_probs[..., class_id]  # (D, H, W) — used only for masking
        for tau in thresholds:
            mask = p > tau
            voxel_count = int(mask.sum())
            mean_kl = float(avg_kl[mask].mean()) if voxel_count > 0 else float("nan")
            rows.append({
                "sample_id":  sample_id,
                "class_id":   class_id,
                "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "threshold":  tau,
                "mean_kl":    mean_kl,
                "voxel_count": voxel_count,
            })
    return rows


def select_threshold(
    all_sweep_rows: list,
    thresholds: list,
    min_voxel_count: int = 10,
) -> float:
    """Pick the threshold with the highest CV of per-structure JSD across all samples.

    Coefficient of variation (CV = std / mean)
    A high CV means the threshold produces well-differentiated per-structure
    estimates rather than just a flat diluted signal. Thresholds where any (sample, class)
    pair has fewer than min_voxel_count voxels are rejected. Falls back to the smallest
    threshold if all of the candidates are rejected.

    Args:
        all_sweep_rows:   combined list of row dicts from compute_threshold_sweep
        thresholds:       ordered list of threshold values
        min_voxel_count:  minimum valid mask size per (sample, class) pair

    Returns the selected threshold (float).
    """
    best_tau = thresholds[0]
    best_cv = -np.inf

    for tau in thresholds:
        rows = [r for r in all_sweep_rows if r["threshold"] == tau]
        # reject if any sample pair has too few voxels
        if any(r["voxel_count"] < min_voxel_count for r in rows):
            continue
        kl_vals = [r["mean_kl"] for r in rows if not np.isnan(r["mean_kl"])]
        if len(kl_vals) < 2:
            continue
        arr = np.array(kl_vals)
        mean_kl = arr.mean()
        if mean_kl == 0.0:
            continue
        cv = arr.std() / mean_kl
        if cv > best_cv:
            best_cv = cv
            best_tau = tau

    return best_tau


def print_threshold_summary(
    all_sweep_rows: list,
    thresholds: list,
    selected_tau: float,
) -> None:
    """Print a per-threshold summary table aggregated over all samples."""

    print("\n--- Threshold Sweep Summary (aggregated over all samples) ---")
    header = f"  {'τ':>7}  {'Mean JSD':>10}  {'Std JSD':>9}  {'CV':>8}  {'Min voxels':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for tau in thresholds:
        rows = [r for r in all_sweep_rows if r["threshold"] == tau]
        kl_vals = [r["mean_kl"] for r in rows if not np.isnan(r["mean_kl"])]
        voxel_counts = [r["voxel_count"] for r in rows]

        if kl_vals:
            arr = np.array(kl_vals)
            mean_kl = arr.mean()
            std_kl = arr.std()
            cv = std_kl / mean_kl if mean_kl > 0 else float("nan")
        else:
            mean_kl = std_kl = cv = float("nan")

        min_v = min(voxel_counts) if voxel_counts else 0
        marker = "  <-- recommended" if tau == selected_tau else ""
        print(f"  {tau:>7.3f}  {mean_kl:>10.6f}  {std_kl:>9.6f}  {cv:>8.4f}  {min_v:>11d}{marker}")

    print(f"\nRecommended threshold: τ = {selected_tau}")
    print(
        "Rationale: maximises coefficient of variation of per-structure JSD\n"
        "           while keeping all (sample, class) soft masks above the minimum voxel count."
    )

# Sanity checks

def validate_stack(stack: np.ndarray, sample_id: int) -> None:
    """Warn if class probabilities don't sum to 1 across any voxel"""
    prob_sum = stack.sum(axis=-1)  # (N, D, H, W)
    mean_sum = prob_sum.mean()
    max_dev = np.abs(prob_sum - 1.0).max()
    if max_dev > 1e-3:
        print(
            f"  [WARN] sample {sample_id}: max deviation from sum-to-1 is "
            f"{max_dev:.4f} (mean {mean_sum:.4f})"
        )



# Visualisation


SEG_CMAP = mcolors.ListedColormap(
    ["black", "#E6194B", "#3CB44B", "#4363D8", "#F58231",
     "#911EB4", "#42D4F4", "#F032E6"]
)
SEG_NORM = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, 8.5), ncolors=8)


def _mid_slice(vol: np.ndarray) -> int:
    return vol.shape[2] // 2


def _axial(vol: np.ndarray, z: int) -> np.ndarray:
    """Return axial slice at index z from (D, H, W) volume."""
    return vol[:, :, z].T  # transpose so H is y-axis in plot


def save_entropy_heatmap(
    entropy: np.ndarray, out_path: Path, title: str = "Predictive Entropy"
) -> None:
    z = _mid_slice(entropy)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(_axial(entropy, z), cmap="hot", origin="lower")
    plt.colorbar(im, ax=ax, label="Entropy (nats)")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_disagreement_heatmap(
    disagreement: np.ndarray, out_path: Path, n_seeds: int, title: str = "Label Disagreement"
) -> None:
    z = _mid_slice(disagreement)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(
        _axial(disagreement, z),
        cmap="YlOrRd",
        origin="lower",
        vmin=1,
        vmax=n_seeds,
    )
    plt.colorbar(im, ax=ax, label="Unique labels across seeds")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_entropy_on_segmentation(
    entropy: np.ndarray,
    consensus: np.ndarray,
    out_path: Path,
    title: str = "Entropy overlay on consensus segmentation",
) -> None:
    z = _mid_slice(entropy)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(
        _axial(consensus.astype(float), z),
        cmap=SEG_CMAP,
        norm=SEG_NORM,
        origin="lower",
        alpha=1.0,
    )
    im = ax.imshow(
        _axial(entropy, z),
        cmap="hot",
        origin="lower",
        alpha=0.6,
        vmin=entropy.min(),
        vmax=entropy.max(),
    )
    plt.colorbar(im, ax=ax, label="Entropy (nats)")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_entropy_on_ct(
    entropy: np.ndarray,
    ct: np.ndarray,
    out_path: Path,
    title: str = "Entropy overlay on CT",
) -> None:
    z = _mid_slice(entropy)
    ct_slice = _axial(ct, z)
    ent_slice = _axial(entropy, z)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(ct_slice, cmap="gray", origin="lower",
              vmin=np.percentile(ct_slice, 2), vmax=np.percentile(ct_slice, 98))
    im = ax.imshow(
        ent_slice,
        cmap="hot",
        origin="lower",
        alpha=0.5,
        vmin=entropy.min(),
        vmax=entropy.max(),
    )
    plt.colorbar(im, ax=ax, label="Entropy (nats)")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# Per-sample processing

def process_sample(
    sample_id: int,
    uncertainty_root: Path,
    test_dir: Path,
    out_dirs: dict,
    seeds: list,
    thresholds: list,
) -> dict:
    """Compute and save all uncertainty outputs for one sample.

    Returns a dictionary with:
        "sweep":    list of threshold-sweep row dicts (aggregated in main for
                    global threshold selection and final per-structure CSV)
        "per_case": dict with global per-case multiclass entropy stats
    """
    print(f"\n--- Sample {sample_id} ---")
    sid = f"sample_{sample_id}"

    
    # Load soft probability maps and argmax segmentations from each seed
    ref_img = None
    soft_maps = []
    argmax_maps = []

    for seed in seeds:
        soft_path = (uncertainty_root / f"seed_{seed}"
                     / f"sample_{sample_id}_warped_atlas_labelmap.nii.gz")
        argmax_path = (uncertainty_root / f"seed_{seed}"
                       / f"sample_{sample_id}_warped_atlas_labelmap_argmax.nii.gz")
        # debug:
        if not soft_path.exists() or not argmax_path.exists():
            raise FileNotFoundError(
                f"Missing file for seed {seed}, sample {sample_id}.\n"
                f"  Expected: {soft_path}\n          {argmax_path}"
            )

        img = nib.load(soft_path)
        if ref_img is None:
            ref_img = img

        soft = load_soft_labelmap(soft_path)   # (D, H, W, C)
        argm = load_argmax(argmax_path)         # (D, H, W)

        soft_maps.append(soft)
        argmax_maps.append(argm)
        print(f"  seed {seed}: soft {soft.shape}, argmax {argm.shape}")
        
    # Next, compute per-seed multiclass entropy maps
    for seed, soft in zip(seeds, soft_maps):
        seed_entropy = compute_entropy(soft)  # (D, H, W)
        save_nifti(
            seed_entropy,
            ref_img,
            out_dirs["per_seed_entropy"] / f"{sid}_seed{seed}_entropy.nii.gz",
        )
    print(f"  Saved: per-seed entropy maps for {len(seeds)} seeds")

    
    # Stack and validate using warning function above
    stack = np.stack(soft_maps, axis=0)           # (N, D, H, W, C)
    argmax_stack = np.stack(argmax_maps, axis=0)  # (N, D, H, W)

    shapes = [s.shape for s in soft_maps]
    assert len(set(shapes)) == 1, f"Shape mismatch across seeds: {shapes}"
    validate_stack(stack, sample_id)

    N = stack.shape[0]
    print(f"  Stack shape: {stack.shape}")

    # compute mean probs, multiclass predictive entropy, consensus argmax
    mean_probs = compute_mean_probs(stack)                       # (D, H, W, C)
    entropy = compute_entropy(mean_probs)                        # (D, H, W)
    # finally the majority vote across all seeds (argmax of mean probabilities)
    consensus = mean_probs.argmax(axis=-1).astype(np.int32)      # (D, H, W)

    # Find per-seed KL divergence from mean
    kl_maps = []
    print(f"  Per-seed KL divergence from mean (mean over all voxels):")
    for seed, soft in zip(seeds, soft_maps):
        kl = compute_kl_divergence(soft, mean_probs)  # (D, H, W)
        kl_maps.append(kl)
        print(f"    seed {seed}: mean KL = {kl.mean():.6f}  max KL = {kl.max():.6f}")
    # Average KL over seeds is actually the Jensen-Shannon Divergence 
    avg_kl = np.mean(np.stack(kl_maps, axis=0), axis=0)  # (D, H, W) == JSD
    print(f"  Average KL (JSD): mean = {avg_kl.mean():.6f}  max = {avg_kl.max():.6f}")

    save_nifti(
        avg_kl,
        ref_img,
        out_dirs["kl_divergence"] / f"{sid}_avg_kl_divergence.nii.gz",
    )
    print(f"  Saved: average KL divergence map")

    # Per-class KL divergence maps — average per-class contribution across seeds
    # Shape: (N, D, H, W, C) → mean over N → (D, H, W, C), then save each class slice
    per_class_kl_stack = np.stack(
        [compute_kl_divergence_per_class(soft, mean_probs) for soft in soft_maps],
        axis=0,
    )  # (N, D, H, W, C)
    avg_per_class_kl = per_class_kl_stack.mean(axis=0)  # (D, H, W, C)
    C = mean_probs.shape[-1]
    for class_id in range(C):
        save_nifti(
            avg_per_class_kl[..., class_id],
            ref_img,
            out_dirs["per_class_kl"] / f"{sid}_class_{class_id}_kl_divergence.nii.gz",
        )
    print(f"  Saved: per-class KL divergence maps for {C} classes")

    # Find the disagreement map
    # count of unique predicted labels across seeds
    disagreement = compute_disagreement(argmax_stack)  # (D, H, W)
    
    # Per-class binary entropy maps - save for all classes (including background)
    # Reminder: for each class c, binary entropy is found where p = mean_probs[..., c].
    for class_id in range(C):
        h = compute_binary_class_entropy(mean_probs[..., class_id])
        save_nifti(
            h,
            ref_img,
            out_dirs["per_class_entropy"] / f"{sid}_class_{class_id}_entropy.nii.gz",
        )
    print(f"  Saved: per-class binary entropy maps for {C} classes")

    # Save the remaining core NIfTI outputs
    save_nifti(
        entropy,
        ref_img,
        out_dirs["entropy"] / f"{sid}_uncertainty_entropy.nii.gz",
    )
    save_nifti(
        mean_probs,
        ref_img,
        out_dirs["mean_preds"] / f"{sid}_mean_warped_atlas_probs.nii.gz",
    )
    save_nifti(
        consensus.astype(np.float32),
        ref_img,
        out_dirs["mean_preds"] / f"{sid}_mean_warped_atlas_argmax.nii.gz",
    )
    save_nifti(
        disagreement.astype(np.float32),
        ref_img,
        out_dirs["disagreement"] / f"{sid}_disagreement_map.nii.gz",
    )
    print(f"  Saved: entropy, mean_probs, mean_argmax, disagreement")

    
    # Next, threshold sweep for per-structure JSD
    # returned to main so global threshold selection can run all samples before
    # writing the final per-structure CSV.
    sweep_rows = compute_threshold_sweep(mean_probs, avg_kl, sample_id, thresholds)
    print(
        f"  Threshold sweep: {len(thresholds)} thresholds × {C - 1} foreground "
        f"classes = {len(sweep_rows)} rows"
    )

    # Global per-case multiclass entropy (kept as a diagnostic alongside the JSD-based per-structure metrics)
    mean_entropy_all = float(entropy.mean())
    fg_mask = consensus > 0
    mean_entropy_fg = float(entropy[fg_mask].mean()) if fg_mask.sum() > 0 else float("nan")
    print(f"  Entropy — all: {mean_entropy_all:.4f}  fg: {mean_entropy_fg:.4f}")

    per_case_row = {
        "sample_id":        sample_id,
        "mean_entropy_all": mean_entropy_all,
        "mean_entropy_fg":  mean_entropy_fg,
    }

    
    # Visualisations
    
    vis_dir = out_dirs["vis"]

    save_entropy_heatmap(
        entropy,
        vis_dir / f"{sid}_entropy_heatmap.png",
        title=f"Sample {sample_id} — Predictive Entropy",
    )
    save_disagreement_heatmap(
        disagreement,
        vis_dir / f"{sid}_disagreement_heatmap.png",
        n_seeds=N,
        title=f"Sample {sample_id} — Label Disagreement (max {N} seeds)",
    )
    save_entropy_on_segmentation(
        entropy,
        consensus,
        vis_dir / f"{sid}_entropy_on_segmentation.png",
        title=f"Sample {sample_id} — Entropy on Consensus Segmentation",
    )

    ct_path = test_dir / f"{sid}_image.nii.gz"
    if ct_path.exists():
        ct = nib.load(ct_path).get_fdata()
        save_entropy_on_ct(
            entropy,
            ct,
            vis_dir / f"{sid}_entropy_on_ct.png",
            title=f"Sample {sample_id} — Entropy on CT",
        )
        print(f"  CT overlay saved.")
    else:
        print(f"  No CT found at {ct_path}, skipping CT overlay.")

    print(f"  Visualisations saved.")
    return {"sweep": sweep_rows, "per_case": per_case_row}



# Main

def discover_sample_ids(uncertainty_root: Path, seeds: list) -> list:
    """Infer sample indices by scanning the first seed folder."""
    first_seed_dir = uncertainty_root / f"seed_{seeds[0]}"
    pattern = re.compile(r"sample_(\d+)_warped_atlas_labelmap\.nii\.gz")
    ids = sorted(
        int(m.group(1))
        for f in first_seed_dir.iterdir()
        if (m := pattern.match(f.name))
    )
    return ids


def validate_seeds(uncertainty_root: Path, seeds: list, sample_ids: list) -> None:
    """Confirm all required files exist across all seeds (before processing starts)."""
    missing = []
    for seed in seeds:
        for sid in sample_ids:
            for suffix in ["warped_atlas_labelmap.nii.gz", "warped_atlas_labelmap_argmax.nii.gz"]:
                p = uncertainty_root / f"seed_{seed}" / f"sample_{sid}_{suffix}"
                if not p.exists():
                    missing.append(str(p))
    # debug 
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} file(s):\n" + "\n".join(missing[:10])
            + ("\n  ..." if len(missing) > 10 else "")
        )


def write_csv(rows: list, fieldnames: list, out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute STN uncertainty maps from multiple Anatomix seeds."
    )
    parser.add_argument(
        "--uncertainty_root",
        default="output/mm-whs/full-stn/uncertainty",
        help="Root folder containing seed_* subdirectories.",
    )
    parser.add_argument(
        "--test_dir",
        default="output/mm-whs/full-stn/test",
        help="Directory containing sample_X_image.nii.gz CT volumes.",
    )
    parser.add_argument(
        "--out_dir",
        default="output/mm-whs/full-stn/uncertainty_analysis",
        help="Root output directory.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seed values to aggregate (default: 42 123 456 789 999).",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=THRESHOLDS,
        help="Probability thresholds for soft-mask sweep (default: 0.001 0.005 0.01 0.05 0.1).",
    )
    args = parser.parse_args()

    uncertainty_root = Path(args.uncertainty_root)
    test_dir = Path(args.test_dir)
    out_root = Path(args.out_dir)

    out_dirs = {
        "entropy":           out_root / "entropy_maps",
        "per_seed_entropy":  out_root / "per_seed_entropy_maps",
        "kl_divergence":     out_root / "kl_divergence_maps",
        "per_class_kl":      out_root / "per_class_kl_maps",
        "disagreement":      out_root / "disagreement_maps",
        "mean_preds":        out_root / "mean_predictions",
        "per_class_entropy": out_root / "per_class_entropy_maps",
        "vis":               out_root / "uncertainty_visualisations",
        "metrics":           out_root / "metrics",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"Uncertainty root: {uncertainty_root}")
    print(f"Output root:      {out_root}")
    print(f"Seeds:            {args.seeds}")
    print(f"Thresholds:       {args.thresholds}")

    sample_ids = discover_sample_ids(uncertainty_root, args.seeds)
    print(f"Found sample IDs: {sample_ids}")
    validate_seeds(uncertainty_root, args.seeds, sample_ids)
    print("File validation passed.")

    all_sweep_rows: list = []
    all_per_case: list = []

    for sample_id in sample_ids:
        result = process_sample(
            sample_id=sample_id,
            uncertainty_root=uncertainty_root,
            test_dir=test_dir,
            out_dirs=out_dirs,
            seeds=args.seeds,
            thresholds=args.thresholds,
        )
        all_sweep_rows.extend(result["sweep"])
        all_per_case.append(result["per_case"])

    
    # Global threshold selection and summary
    selected_tau = select_threshold(all_sweep_rows, args.thresholds)
    print_threshold_summary(all_sweep_rows, args.thresholds, selected_tau)

    
    # Write output CSVs
    sweep_csv  = out_dirs["metrics"] / "per_structure_uncertainty_threshold_sweep.csv"
    struct_csv = out_dirs["metrics"] / "per_structure_uncertainty.csv"
    case_csv   = out_dirs["metrics"] / "per_case_uncertainty.csv"

    # Full threshold sweep — all thresholds, useful for downstream sensitivity analysis
    write_csv(
        all_sweep_rows,
        ["sample_id", "class_id", "class_name", "threshold", "mean_kl", "voxel_count"],
        sweep_csv,
    )

    # Final per-structure metrics at the automatically selected threshold
    final_per_structure = [r for r in all_sweep_rows if r["threshold"] == selected_tau]
    write_csv(
        final_per_structure,
        ["sample_id", "class_id", "class_name", "threshold", "mean_kl", "voxel_count"],
        struct_csv,
    )

    # Per-case global multiclass entropy 
    write_csv(
        all_per_case,
        ["sample_id", "mean_entropy_all", "mean_entropy_fg"],
        case_csv,
    )

    print(f"\nMetrics saved:")
    print(f"  {sweep_csv}  (full threshold sweep)")
    print(f"  {struct_csv}  (final metrics at τ = {selected_tau})")
    print(f"  {case_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()

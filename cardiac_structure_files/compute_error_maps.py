"""
Compute voxel-wise error maps between consensus segmentation and ground truth
for the MM-WHS test set.

For each test sample:
  - Multi-class error map:   (GT != consensus)  (1 wherever labels differ, 0 where correct)
  - Per-structure error maps: for each class c, |binary_GT_c - binary_consensus_c|
                              (marks false positives and false negatives for that structure)

Consensus segmentation:  uncertainty_analysis/mean_predictions/sample_X_mean_warped_atlas_argmax.nii.gz
Ground truth:            test/sample_X_labelmap_argmax.nii.gz  (preprocessed, same 96^3 space)

Output directory: uncertainty_analysis/error_maps/
  sample_X_error_multiclass.nii.gz
  sample_X_error_class_Y.nii.gz   (one per foreground structure, Y=1..7)
"""

import re
from pathlib import Path

import nibabel as nib
import numpy as np

CLASS_NAMES = {
    0: "Background",
    1: "Myocardium",
    2: "Left_Atrium",
    3: "Left_Ventricle",
    4: "Right_Atrium",
    5: "Right_Ventricle",
    6: "Aorta",
    7: "Pulmonary_Artery",
}

MEAN_PREDS_DIR = Path("output/mm-whs/full-stn/uncertainty_analysis/mean_predictions")
TEST_DIR       = Path("output/mm-whs/full-stn/test")
OUT_DIR        = Path("output/mm-whs/full-stn/uncertainty_analysis/error_maps")


def save_nifti(data: np.ndarray, ref: nib.Nifti1Image, path: Path) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.float32), ref.affine, ref.header), str(path))


def compute_error_maps(sample_id: int) -> None:
    sid = f"sample_{sample_id}"

    consensus_path = MEAN_PREDS_DIR / f"{sid}_mean_warped_atlas_argmax.nii.gz"
    gt_path        = TEST_DIR       / f"{sid}_labelmap_argmax.nii.gz"

    ref      = nib.load(consensus_path)
    consensus = ref.get_fdata().astype(np.int32)
    gt        = nib.load(gt_path).get_fdata().astype(np.int32)

    assert consensus.shape == gt.shape, (
        f"Shape mismatch for sample {sample_id}: {consensus.shape} vs {gt.shape}"
    )

    # Multi-class binary error: 1 wherever predicted label != GT label, 0 where correct
    multiclass_error = (gt != consensus).astype(np.float32)
    save_nifti(multiclass_error, ref, OUT_DIR / f"{sid}_error_multiclass.nii.gz")

    # Per-structure binary error: |binary_GT_c - binary_consensus_c|
    # Gives 1 at false positives (consensus=c but GT!=c) and false negatives (GT=c but consensus!=c)
    n_classes = max(CLASS_NAMES.keys()) + 1
    for class_id in range(1, n_classes):  # skip background
        gt_binary        = (gt == class_id).astype(np.float32)
        consensus_binary = (consensus == class_id).astype(np.float32)
        structure_error  = np.abs(gt_binary - consensus_binary)
        save_nifti(
            structure_error, ref,
            OUT_DIR / f"{sid}_error_class_{class_id}_{CLASS_NAMES[class_id]}.nii.gz",
        )

    n_voxels = gt.size
    n_errors = int((gt != consensus).sum())
    print(f"  sample {sample_id}: {n_errors}/{n_voxels} voxels wrong "
          f"({100 * n_errors / n_voxels:.2f}%)")


def discover_sample_ids() -> list:
    pattern = re.compile(r"sample_(\d+)_mean_warped_atlas_argmax\.nii\.gz")
    return sorted(
        int(m.group(1))
        for f in MEAN_PREDS_DIR.iterdir()
        if (m := pattern.match(f.name))
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_ids = discover_sample_ids()
    print(f"Found {len(sample_ids)} test samples: {sample_ids}")
    print(f"Output directory: {OUT_DIR}\n")

    for sid in sample_ids:
        compute_error_maps(sid)

    print(f"\nDone. Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

#!/bin/bash
# Compute uncertainty bundle features for all 5 CV folds in parallel.
#
# For each fold, runs TWO passes:
#   Pass A — test  split (8 patients)  → full-stn/uncertainty_analysis/metrics/uncertainty_bundle_features.csv
#   Pass B — train split (32 patients) → full-stn-train-unc/uncertainty_analysis/metrics/uncertainty_bundle_features.csv
#
# Features computed per subject:
#   def_variance               — mean voxelwise variance magnitude of 5 STN transform fields
#   volume_variance            — variance of coronary mask voxel count across 5 seeds
#   pairwise_dice_disagreement — mean pairwise (1 – Dice) across all 10 seed pairs
#
# Reads from existing uncertainty inference outputs (run_coronary_uncertainty_seeds.sh):
#   output/asoca-coronary/fold_k/full-stn/uncertainty/seed_X/sample_N_transform.nii.gz
#   output/asoca-coronary/fold_k/full-stn/uncertainty/seed_X/sample_N_image_prime_argmax.nii.gz
#   (and the equivalent train-unc paths)
#
# Does NOT rerun the STN. Does NOT overwrite any existing CSVs.
# Only writes uncertainty_bundle_features.csv (new unique filename).
#
# Prerequisites:
#   run_coronary_seg_finetune.sh      — segmentation checkpoints
#   run_coronary_atlas_istn.sh        — trained fold-specific STNs
#   run_coronary_uncertainty_seeds.sh — 5-seed transform fields + argmax labelmaps (must be complete)
#
# Usage:
#   sbatch run_coronary_compute_unc_bundle.sh

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=coronary_unc_bundle
#SBATCH --output=logs/experiment_unc_bundle/coronary_unc_bundle_fold_%a_%j.log

FOLD=fold_${SLURM_ARRAY_TASK_ID}

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

mkdir -p logs/experiment_unc_bundle

echo "========================================"
echo "Fold : ${FOLD}"
echo "Date : $(date)"
echo "Host : $(hostname)"
echo "========================================"

# ── Pass A: test split ────────────────────────────────────────────────────────
echo ""
echo "--- Pass A: test split (8 patients) ---"
echo "    Input : output/asoca-coronary/${FOLD}/full-stn/uncertainty/"
echo "    Output: output/asoca-coronary/${FOLD}/full-stn/uncertainty_analysis/metrics/uncertainty_bundle_features.csv"

python -u coronary-compute-unc-bundle.py \
    --uncertainty_root "output/asoca-coronary/${FOLD}/full-stn/uncertainty" \
    --out_dir          "output/asoca-coronary/${FOLD}/full-stn/uncertainty_analysis"

echo ""
echo "Pass A complete."

# ── Pass B: train split ───────────────────────────────────────────────────────
echo ""
echo "--- Pass B: train split (32 patients) ---"
echo "    Input : output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty/"
echo "    Output: output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty_analysis/metrics/uncertainty_bundle_features.csv"

python -u coronary-compute-unc-bundle.py \
    --uncertainty_root "output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty" \
    --out_dir          "output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty_analysis"

echo ""
echo "Pass B complete."

echo ""
echo "========================================"
echo "Fold ${FOLD} — uncertainty bundle features written."
echo "Date : $(date)"
echo "========================================"

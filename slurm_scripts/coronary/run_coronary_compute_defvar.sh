#!/bin/bash
# Compute deformation_variance_mean for all 5 CV folds in parallel.
#
# For each fold, runs TWO passes:
#   Pass A — test  split (8 patients)  → full-stn/uncertainty_analysis/metrics/deformation_variance_summary.csv
#   Pass B — train split (32 patients) → full-stn-train-unc/uncertainty_analysis/metrics/deformation_variance_summary.csv
#
# Reads from existing transform fields (already computed by run_coronary_uncertainty_seeds.sh):
#   output/asoca-coronary/fold_k/full-stn/uncertainty/seed_X/sample_N_transform.nii.gz
#   output/asoca-coronary/fold_k/full-stn-train-unc/uncertainty/seed_X/sample_N_transform.nii.gz
#
# Does NOT rerun the STN. Does NOT overwrite any entropy-based CSVs.
# Only writes deformation_variance_summary.csv (new unique filename).
#
# Prerequisites:
#   run_coronary_seg_finetune.sh     — segmentation checkpoints
#   run_coronary_atlas_istn.sh       — trained fold-specific STNs
#   run_coronary_uncertainty_seeds.sh — 5-seed transform fields (must be complete)
#
# Usage:
#   sbatch run_coronary_compute_defvar.sh

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=coronary_defvar
#SBATCH --output=logs/experiment_defvar/coronary_defvar_fold_%a_%j.log

FOLD=fold_${SLURM_ARRAY_TASK_ID}

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

mkdir -p logs/experiment_defvar

echo "========================================"
echo "Fold : ${FOLD}"
echo "Date : $(date)"
echo "Host : $(hostname)"
echo "========================================"

# ── Pass A: test split ────────────────────────────────────────────────────────
echo ""
echo "--- Pass A: test split (8 patients) ---"
echo "    Input : output/asoca-coronary/${FOLD}/full-stn/uncertainty/"
echo "    Output: output/asoca-coronary/${FOLD}/full-stn/uncertainty_analysis/metrics/deformation_variance_summary.csv"

python -u coronary-compute-defvar.py \
    --uncertainty_root "output/asoca-coronary/${FOLD}/full-stn/uncertainty" \
    --out_dir          "output/asoca-coronary/${FOLD}/full-stn/uncertainty_analysis"

echo ""
echo "Pass A complete."

# ── Pass B: train split ───────────────────────────────────────────────────────
echo ""
echo "--- Pass B: train split (32 patients) ---"
echo "    Input : output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty/"
echo "    Output: output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty_analysis/metrics/deformation_variance_summary.csv"

python -u coronary-compute-defvar.py \
    --uncertainty_root "output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty" \
    --out_dir          "output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty_analysis"

echo ""
echo "Pass B complete."

echo ""
echo "========================================"
echo "Fold ${FOLD} — deformation variance summary written."
echo "Date : $(date)"
echo "========================================"

#!/bin/bash
# Compute per-sample uncertainty metrics from multi-seed STN outputs.
# Pure numpy/nibabel post-processing — no GPU computation.
#
# Each array task handles one fold and runs two sequential calls:
#   Call 1 — test split  (8 patients)  reads from full-stn/uncertainty/
#   Call 2 — train split (32 patients) reads from full-stn-train-unc/uncertainty/
#
# Each call writes to uncertainty_analysis/metrics/:
#   per_structure_uncertainty.csv      (N rows × 1 class, at auto-selected τ)
#   per_case_uncertainty.csv           (N rows: mean_entropy_all, mean_entropy_fg)
#   per_structure_uncertainty_threshold_sweep.csv
#
# Must run after run_coronary_uncertainty_seeds.sh is complete for all 5 folds.
#
# Usage:
#   sbatch run_coronary_compute_uncertainty.sh

#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=coronary_unc_compute
#SBATCH --output=logs/experiment_4/coronary_unc_compute_fold_%a_%j.log

FOLD=fold_${SLURM_ARRAY_TASK_ID}

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

echo "========================================"
echo "Fold : ${FOLD}"
echo "Date : $(date)"
echo "Host : $(hostname)"
echo "========================================"

# ── Call 1: test split (8 patients) ──────────────────────────────────────────
# Reads:   output/asoca-coronary/${FOLD}/full-stn/uncertainty/seed_X/sample_{0..7}_*.nii.gz
# Writes:  output/asoca-coronary/${FOLD}/full-stn/uncertainty_analysis/metrics/
#          per_structure_uncertainty.csv  (8 rows — matches test dataloader sample_idx 0-7)
#          per_case_uncertainty.csv       (8 rows)

echo ""
echo "--- Call 1: test split (8 patients) ---"
echo "    Input : output/asoca-coronary/${FOLD}/full-stn/uncertainty/"
echo "    Output: output/asoca-coronary/${FOLD}/full-stn/uncertainty_analysis/"
echo ""

python -u coronary-compute-uncertainty.py \
    --uncertainty_root "output/asoca-coronary/${FOLD}/full-stn/uncertainty" \
    --out_dir          "output/asoca-coronary/${FOLD}/full-stn/uncertainty_analysis"

echo ""
echo "Call 1 complete."

# ── Call 2: train split (32 patients) ────────────────────────────────────────
# Reads:   output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty/seed_X/sample_{0..31}_*.nii.gz
# Writes:  output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty_analysis/metrics/
#          per_structure_uncertainty.csv  (32 rows — matches train dataloader sample_idx 0-31)
#          per_case_uncertainty.csv       (32 rows)

echo ""
echo "--- Call 2: train split (32 patients) ---"
echo "    Input : output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty/"
echo "    Output: output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty_analysis/"
echo ""

python -u coronary-compute-uncertainty.py \
    --uncertainty_root "output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty" \
    --out_dir          "output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty_analysis"

echo ""
echo "Call 2 complete."
echo ""
echo "========================================"
echo "Fold ${FOLD} — uncertainty computation done."
echo "Date : $(date)"
echo "========================================"

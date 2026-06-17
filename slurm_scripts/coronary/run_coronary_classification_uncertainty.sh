#!/bin/bash
# Run coronary uncertainty classification for all 5 CV folds in parallel.
# Reads uncertainty features from:
#   output/asoca-coronary/fold_k/full-stn/uncertainty_analysis/metrics/per_structure_uncertainty.csv
#   output/asoca-coronary/fold_k/full-stn-train-unc/uncertainty_analysis/metrics/per_structure_uncertainty.csv
# Writes to output/asoca-coronary/fold_k/classification_uncertainty/ (does NOT touch classification/).
#
# Prerequisites:
#   run_coronary_uncertainty_seeds.sh    — must be complete for all 5 folds
#   run_coronary_compute_uncertainty.sh  — must be complete for all 5 folds
#
# Usage:
#   sbatch run_coronary_classification_uncertainty.sh

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=coronary_cls_unc
#SBATCH --output=logs/experiment_4/coronary_cls_unc_fold_%a_%j.log

FOLD=fold_${SLURM_ARRAY_TASK_ID}

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

echo "========================================"
echo "Fold : ${FOLD}"
echo "Date : $(date)"
echo "Host : $(hostname)"
echo "========================================"

python -u coronary-classification-uncertainty.py --fold "${FOLD}" --use-uncertainty

echo "========================================"
echo "Fold ${FOLD} — uncertainty classification done."
echo "Date : $(date)"
echo "========================================"

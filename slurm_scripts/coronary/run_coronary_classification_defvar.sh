#!/bin/bash
# Run Baseline + DeformationVariance classification for all 5 CV folds in parallel.
#
# Experiment: 110 baseline features (3 geometric + 107 radiomics)
#           + 1 deformation uncertainty feature (deformation_variance_mean)
#           = 111 total features
#
# All outputs go to:
#   output/asoca-coronary/fold_k/classification_defvar/
#     features_defvar.csv
#     predictions_defvar.csv
#     metrics_defvar.json
#     results_defvar.csv
#     mlp_defvar.pt
#
# NOTHING from classification/, classification_uncertainty/, or any entropy
# experiment is touched.
#
# Prerequisites (all must be complete before submitting):
#   run_coronary_seg_finetune.sh      — 25 segmentation checkpoints
#   run_coronary_atlas_istn.sh        — 5 trained fold-specific STNs + atlases
#   run_coronary_uncertainty_seeds.sh — 5-seed transform fields
#   run_coronary_compute_defvar.sh    — deformation_variance_summary.csv (both splits, all folds)
#
# Usage:
#   sbatch run_coronary_classification_defvar.sh

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=coronary_cls_defvar
#SBATCH --output=logs/experiment_defvar/coronary_cls_defvar_fold_%a_%j.log

FOLD=fold_${SLURM_ARRAY_TASK_ID}

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

mkdir -p logs/experiment_defvar

echo "========================================"
echo "Fold         : ${FOLD}"
echo "Experiment   : Baseline + DeformationVariance"
echo "Output dir   : output/asoca-coronary/${FOLD}/classification_defvar/"
echo "Date         : $(date)"
echo "Host         : $(hostname)"
echo "========================================"

python -u coronary-classification-defvar.py --fold "${FOLD}"

echo ""
echo "========================================"
echo "Fold ${FOLD} — classification_defvar complete."
echo "Date : $(date)"
echo "========================================"

#!/bin/bash
# Run Baseline + Uncertainty Bundle classification for all 5 CV folds in parallel.
#
# Experiment: 110 baseline features (3 geometric + 107 radiomics)
#           + 3 uncertainty bundle features:
#               def_variance, volume_variance, pairwise_dice_disagreement
#           = 113 total features (at max def_pc_amt=3; Optuna searches 1–3)
#
# All outputs go to:
#   output/asoca-coronary/fold_k/classification_unc_bundle/
#     features_unc_bundle.csv
#     predictions_unc_bundle.csv
#     metrics_unc_bundle.json
#     results_unc_bundle.csv
#     mlp_unc_bundle.pt
#
# NOTHING from classification/, classification_uncertainty/, classification_defvar/,
# or any other experiment is touched.
#
# Prerequisites (all must be complete before submitting):
#   run_coronary_seg_finetune.sh         — 25 segmentation checkpoints
#   run_coronary_atlas_istn.sh           — 5 trained fold-specific STNs + atlases
#   run_coronary_uncertainty_seeds.sh    — 5-seed transform fields + argmax labelmaps
#   run_coronary_compute_unc_bundle.sh   — uncertainty_bundle_features.csv (both splits, all folds)
#
# Usage:
#   sbatch run_coronary_classification_unc_bundle.sh

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=coronary_cls_unc_bundle
#SBATCH --output=logs/experiment_unc_bundle/coronary_cls_unc_bundle_fold_%a_%j.log

FOLD=fold_${SLURM_ARRAY_TASK_ID}

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

mkdir -p logs/experiment_unc_bundle

echo "========================================"
echo "Fold         : ${FOLD}"
echo "Experiment   : Baseline + Uncertainty Bundle"
echo "Output dir   : output/asoca-coronary/${FOLD}/classification_unc_bundle/"
echo "Date         : $(date)"
echo "Host         : $(hostname)"
echo "========================================"

python -u coronary-classification-unc-bundle.py --fold "${FOLD}"

echo ""
echo "========================================"
echo "Fold ${FOLD} — classification_unc_bundle complete."
echo "Date : $(date)"
echo "========================================"

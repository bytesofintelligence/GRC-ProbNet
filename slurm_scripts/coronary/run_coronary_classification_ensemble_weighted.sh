#!/bin/bash
# Run EnsembleWeighted classification for all 5 CV folds in parallel.
#
# Experiment: ensemble_weighted
#   110 features (3 geometric + 107 radiomics), fused via uncertainty-aware weights.
#   Weights derived from per-seed geometry disagreement (||G_i - G_mean||^2 → exp(-u_i)).
#   No explicit uncertainty scalars — uncertainty drives feature weighting, not appended dims.
#
# All outputs go to:
#   output/asoca-coronary/fold_k/classification_ensemble_weighted/
#     features.csv
#     predictions.csv
#     metrics.json
#     results_ensemble_weighted.csv
#     mlp.pt
#
# NOTHING from classification/, classification_uncertainty/, classification_defvar/,
# classification_unc_bundle/, or classification_ensemble_mean_only/ is touched.
#
# Prerequisites (all must be complete before submitting):
#   run_coronary_seg_finetune.sh          — per-fold baseline checkpoints
#   run_coronary_atlas_istn.sh            — 5 trained fold-specific STNs + atlases
#   run_coronary_uncertainty_seeds.sh     — per-fold seed-specific Anatomix checkpoints
#                                           (saved_models/segmentation_asoca/fold_k/seed_X/)
#
# Usage:
#   sbatch run_coronary_classification_ensemble_weighted.sh

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=coronary_cls_ens_w
#SBATCH --output=logs/experiment_ensemble_weighted/coronary_cls_ens_w_fold_%a_%j.log

FOLD=fold_${SLURM_ARRAY_TASK_ID}

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

mkdir -p logs/experiment_ensemble_weighted

echo "========================================"
echo "Fold         : ${FOLD}"
echo "Experiment   : EnsembleWeighted (5 seeds, 110 features, uncertainty-weighted fusion)"
echo "Output dir   : output/asoca-coronary/${FOLD}/classification_ensemble_weighted/"
echo "Date         : $(date)"
echo "Host         : $(hostname)"
echo "========================================"

python -u coronary-classification-uncertainty-ensemble-weighted.py --fold "${FOLD}"

echo ""
echo "========================================"
echo "Fold ${FOLD} — classification_ensemble_weighted complete."
echo "Date : $(date)"
echo "========================================"

#!/bin/bash
# Run EnsembleMeanOnly classification for all 5 CV folds in parallel.
#
# Experiment: ensemble_mean_only
#   110 features (3 geometric + 107 radiomics), each the mean across 5 Anatomix seeds.
#   No explicit uncertainty scalars — segmentation uncertainty is propagated through
#   feature averaging (ensemble of 5 independently fine-tuned Anatomix checkpoints).
#
# All outputs go to:
#   output/asoca-coronary/fold_k/classification_ensemble_mean_only/
#     features.csv
#     predictions.csv
#     metrics.json
#     results_ensemble_mean_only.csv
#     mlp.pt
#
# NOTHING from classification/, classification_uncertainty/, classification_defvar/,
# or classification_unc_bundle/ is touched.
#
# Prerequisites (all must be complete before submitting):
#   run_coronary_seg_finetune.sh          — per-fold baseline checkpoints
#   run_coronary_atlas_istn.sh            — 5 trained fold-specific STNs + atlases
#   run_coronary_uncertainty_seeds.sh     — per-fold seed-specific Anatomix checkpoints
#                                           (saved_models/segmentation_asoca/fold_k/seed_X/)
#
# Usage:
#   sbatch run_coronary_classification_ensemble_mean_only.sh

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=coronary_cls_ens
#SBATCH --output=logs/experiment_ensemble_mean_only/coronary_cls_ens_fold_%a_%j.log

FOLD=fold_${SLURM_ARRAY_TASK_ID}

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

mkdir -p logs/experiment_ensemble_mean_only

echo "========================================"
echo "Fold         : ${FOLD}"
echo "Experiment   : EnsembleMeanOnly (5 seeds, 110 features)"
echo "Output dir   : output/asoca-coronary/${FOLD}/classification_ensemble_mean_only/"
echo "Date         : $(date)"
echo "Host         : $(hostname)"
echo "========================================"

python -u coronary-classification-uncertainty-ensemble-mean-only.py --fold "${FOLD}"

echo ""
echo "========================================"
echo "Fold ${FOLD} — classification_ensemble_mean_only complete."
echo "Date : $(date)"
echo "========================================"

#!/bin/bash
# Run EnsembleMeanStd classification for all 5 CV folds in parallel.
#
# Experiment: ensemble_mean_std
#   220 features per subject: [F_mean || F_std] across 5 Anatomix seeds.
#     110 mean features (3 geometric + 107 radiomics, averaged over seeds)
#     110 std  features (3 geometric + 107 radiomics, std-dev  over seeds)
#   SelectKBest(f_classif, k=K) fitted ONLY on the training fold inside Optuna HPO.
#   K is a tuned hyperparameter; test fold is never seen during selection.
#
# All outputs go to:
#   output/asoca-coronary/fold_k/classification_ensemble_mean_std/
#     features.csv          (220-D, all 40 patients)
#     predictions.csv       (test-fold patients)
#     metrics.json
#     results_ensemble_mean_std.csv
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
#   sbatch run_coronary_classification_ensemble_mean_std.sh

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=coronary_cls_ms
#SBATCH --output=logs/experiment_ensemble_mean_std/coronary_cls_ms_fold_%a_%j.log

FOLD=fold_${SLURM_ARRAY_TASK_ID}

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

mkdir -p logs/experiment_ensemble_mean_std

echo "========================================"
echo "Fold         : ${FOLD}"
echo "Experiment   : EnsembleMeanStd (5 seeds, 220 features, SelectKBest)"
echo "Output dir   : output/asoca-coronary/${FOLD}/classification_ensemble_mean_std/"
echo "Date         : $(date)"
echo "Host         : $(hostname)"
echo "========================================"

python -u coronary-classification-uncertainty-ensemble-mean-std.py --fold "${FOLD}"

echo ""
echo "========================================"
echo "Fold ${FOLD} — classification_ensemble_mean_std complete."
echo "Date : $(date)"
echo "========================================"

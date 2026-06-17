#!/bin/bash

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=classify_ensemble

# Experiment 3 — Prediction-space uncertainty ensemble.
#
# Prerequisites (must complete first):
#   1. run_asoca_uncertainty_seeds.sh
#      → generates per-seed warped-atlas outputs under output/asoca/uncertainty/seed_*/
#      → specifically needs:
#            sample_<N>_warped_atlas_labelmap_argmax.nii.gz   (labelmap for radiomics)
#            sample_<N>_transform.nii.gz                      (STN warp for geometry)
#
# Does NOT require run_compute_uncertainty_asoca.sh.
# Does NOT require retraining the STN or Anatomix.
#
# What this does differently from Experiment 1 (baseline):
#   Training : identical — single consensus segmentation, same MLP + Optuna.
#   Inference: each of the 5 per-seed segmentations is independently passed
#              through the trained classifier → 5 probabilities → aggregated.
#
# Outputs:
#   results_prediction_ensemble.csv          (fold-level metrics)
#   results_prediction_ensemble_subjects.csv (per-subject probabilities + entropy)

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u Geo-Radio-Classification-PredictionEnsemble.py \
    --uncertainty-root output/asoca/uncertainty \
    --seg-seeds 42 123 456 789 999 \
    "$@"

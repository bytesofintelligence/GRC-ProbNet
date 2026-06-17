#!/bin/bash

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=classify_multiseed

# Experiment 2 — Multi-seed radiomic uncertainty aggregation.
#
# Prerequisites (must complete first):
#   1. run_asoca_uncertainty_seeds.sh   → generates per-seed warped-atlas outputs
#      (output/asoca/uncertainty/seed_*/sample_*_warped_atlas_labelmap_argmax.nii.gz)
#
# Does NOT require run_compute_uncertainty_asoca.sh — Experiment 2 does not use
# entropy maps.  No atlas-istn-anatomix rerun is needed.
#
# Outputs:
#   results_multiseed_radio.csv     (radiomics-only, 1498 features)
#
# To also include geometric features (1540 features) run with --use-multiseed-geometry:
#   sbatch run_classification_multiseed.sh --use-multiseed-geometry
#   → output: results_multiseed_geo_radio.csv

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u Geo-Radio-Classification-MultiSeed.py \
    --uncertainty-root output/asoca/uncertainty \
    --seeds 42 123 456 789 999 \
    "$@"

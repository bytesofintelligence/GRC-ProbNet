#!/bin/bash

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=classify_uncertainty

# Job 4 of 4 — Uncertainty-augmented geo-radio classification.
#
# USE_UNCERTAINTY_FEATURES=True (--use-uncertainty flag): the 7 unc_* columns
# are appended to selected_cols in every Optuna trial, so the MLP sees them.
# After Optuna, runs first-layer weight + permutation importance analysis.
#
# Results saved to: results_uncertainty.csv
#
# Must run after Job 3 (run_classification_baseline.sh) to avoid GPU contention.

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u Geo-Radio-Classification.py \
    --use-uncertainty \
    --uncertainty-csv output/asoca/uncertainty_analysis/metrics/per_structure_uncertainty.csv

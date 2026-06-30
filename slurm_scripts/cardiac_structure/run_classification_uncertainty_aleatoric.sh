#!/bin/bash

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=classify_unc_aleat

# Uncertainty-augmented geo-radio classification using per-structure aleatoric features.
#
# USE_UNCERTAINTY_FEATURES=True (--use-uncertainty flag): 7 unc_* columns
# (per-structure aleatoric = H(mean_p) − JSD from compute_uncertainty_aleatoric.py) are
# appended to selected_cols in every Optuna trial, alongside geometric and radiomic features.
# After Optuna, runs first-layer weight + permutation importance analysis.
#
# Results saved to: results_uncertainty_aleatoric.csv
#
# Must run after run_compute_uncertainty_asoca_aleatoric.sh.

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u cardiac_structure_files/Geo-Radio-Classification_aleatoric.py \
    --use-uncertainty \
    --uncertainty-csv output/asoca/uncertainty_analysis_aleatoric/metrics/per_structure_uncertainty.csv

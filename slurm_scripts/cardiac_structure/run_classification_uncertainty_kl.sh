#!/bin/bash

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=classify_unc_kl

# Uncertainty-augmented geo-radio classification using JSD features (KL variant).
#
# USE_UNCERTAINTY_FEATURES=True (--use-uncertainty flag): the 7 unc_* columns
# (per-structure masked mean JSD from compute_uncertainty_kl.py) are appended to
# selected_cols in every Optuna trial, so the MLP sees them alongside the
# geometric and radiomic features.
# After Optuna, runs first-layer weight + permutation importance analysis.
#
# Results saved to: results_uncertainty_kl.csv
#
# Must run after run_compute_uncertainty_asoca_kl.sh.

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u cardiac_structure_files/Geo-Radio-Classification_kl.py \
    --use-uncertainty \
    --uncertainty-csv output/asoca/uncertainty_analysis_kl/metrics/per_structure_uncertainty.csv

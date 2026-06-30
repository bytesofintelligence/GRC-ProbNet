#!/bin/bash

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=classify_unc_aleat1

# Uncertainty-augmented geo-radio classification using a single global aleatoric feature.
#
# USE_UNCERTAINTY_FEATURES=True (--use-uncertainty flag): 1 unc_aleatoric column
# (global foreground mean of H(mean_p) − JSD from compute_uncertainty_aleatoric_only_one.py) is
# appended to selected_cols in every Optuna trial, alongside geometric and radiomic features.
# After Optuna, runs first-layer weight + permutation importance analysis.
#
# Results saved to: results_uncertainty_aleatoric_one.csv
#
# Must run after run_compute_uncertainty_asoca_aleatoric_only_one.sh.

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u cardiac_structure_files/Geo-Radio-Classification_aleatoric_only_one.py \
    --use-uncertainty \
    --uncertainty-csv output/asoca/uncertainty_analysis_aleatoric_only_one/metrics/per_case_uncertainty.csv

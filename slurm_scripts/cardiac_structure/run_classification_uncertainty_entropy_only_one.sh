#!/bin/bash

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=classify_unc_ent_one

# Uncertainty-augmented geo-radio classification using ONE global entropy feature per image.
#
# USE_UNCERTAINTY_FEATURES=True (--use-uncertainty flag): adds a single "unc_entropy" column
# (mean foreground predictive entropy from compute_uncertainty-entropy_only_one.py) to
# selected_cols in every Optuna trial, alongside geometric and radiomic features.
# After Optuna, runs first-layer weight + permutation importance analysis.
#
# Results saved to: results_uncertainty_entropy_one.csv
#
# Must run after run_compute_uncertainty_asoca_entropy_only_one.sh.

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u cardiac_structure_files/Geo-Radio-Classification_entropy_only_one.py \
    --use-uncertainty \
    --uncertainty-csv output/asoca/uncertainty_analysis_entropy_only_one/metrics/per_case_uncertainty.csv

#!/bin/bash

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=classify_baseline

# Job 3 of 4 — Baseline geo-radio classification (radiomic + geometric features only).
#
# USE_UNCERTAINTY_FEATURES=False (default): the 7 unc_* columns are loaded into
# df_full but never passed to the MLP, so the classifier is unaware of them.
#
# Results saved to: results_baseline.csv
#
# Must run after Job 2 (run_compute_uncertainty_asoca.sh), because the uncertainty
# CSV is always loaded at startup even in the baseline run.

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u Geo-Radio-Classification.py \
    --uncertainty-csv output/asoca/uncertainty_analysis/metrics/per_structure_uncertainty.csv

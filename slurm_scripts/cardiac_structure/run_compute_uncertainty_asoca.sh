#!/bin/bash

#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=asoca_unc_compute

# Job 2 of 4 — Compute per-structure uncertainty maps from ASOCA multi-seed outputs.
#
# Reads from:  output/asoca/uncertainty/seed_{42,123,456,789,999}/sample_N_*.nii.gz
# Writes to:   output/asoca/uncertainty_analysis/
#
# Produces the key file consumed by Jobs 3 & 4:
#   output/asoca/uncertainty_analysis/metrics/per_structure_uncertainty.csv
#   (39 subjects × 7 structures = 273 rows)
#
# Must run after Job 1 (run_asoca_uncertainty_seeds.sh).

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u compute_uncertainty.py \
    --uncertainty_root output/asoca/uncertainty \
    --out_dir          output/asoca/uncertainty_analysis

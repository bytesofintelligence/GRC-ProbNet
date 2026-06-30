#!/bin/bash

#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=asoca_unc_kl

# Compute per-structure JSD uncertainty maps from ASOCA multi-seed outputs.
#
# Reads from:  output/asoca/uncertainty/seed_{42,123,456,789,999}/sample_N_*.nii.gz
#              (same seed outputs as run_compute_uncertainty_asoca.sh — no re-running seeds needed)
# Writes to:   output/asoca/uncertainty_analysis_kl/
#
# Produces the key file consumed by run_classification_uncertainty_kl.sh:
#   output/asoca/uncertainty_analysis_kl/metrics/per_structure_uncertainty.csv
#   (39 subjects × 7 structures = 273 rows, with mean_kl column instead of mean_entropy)
#
# Must run after run_asoca_uncertainty_seeds.sh.

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u cardiac_structure_files/compute_uncertainty_kl.py \
    --uncertainty_root output/asoca/uncertainty \
    --out_dir          output/asoca/uncertainty_analysis_kl

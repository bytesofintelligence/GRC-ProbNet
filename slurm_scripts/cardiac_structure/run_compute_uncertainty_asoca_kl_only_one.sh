#!/bin/bash

#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=asoca_unc_kl_one

# Compute ONE global JSD value per image from ASOCA multi-seed outputs.
# No per-structure masking or threshold sweep — just mean JSD over foreground voxels.
#
# Reads from:  output/asoca/uncertainty/seed_{42,123,456,789,999}/sample_N_*.nii.gz
#              (same seed outputs as run_asoca_uncertainty_seeds.sh — no re-running needed)
# Writes to:   output/asoca/uncertainty_analysis_kl_only_one/
#
# Produces the key file consumed by run_classification_uncertainty_kl_only_one.sh:
#   output/asoca/uncertainty_analysis_kl_only_one/metrics/per_case_uncertainty.csv
#   (39 rows × 3 cols: sample_id, mean_kl, voxel_count)
#
# Must run after run_asoca_uncertainty_seeds.sh.

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u cardiac_structure_files/compute_uncertainty_kl_only_one.py \
    --uncertainty_root output/asoca/uncertainty \
    --out_dir          output/asoca/uncertainty_analysis_kl_only_one

#!/bin/bash

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=asoca_unc_seeds

# Job 1 of 4 — Generate ASOCA multi-seed warped atlas outputs.
#
# Runs atlas-istn-anatomix.py in uncertainty mode on the ASOCA inference set
# (data/config/inference.csv, 39 subjects) using 5 Anatomix seeds.
#
# Outputs go to output/asoca/uncertainty/seed_<X>/sample_<N>_*.nii.gz
# (separate from the MM-WHS uncertainty outputs in output/mm-whs/full-stn/uncertainty/).
#
# Must complete before Job 2 (run_compute_uncertainty_asoca.sh).

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u atlas-istn-anatomix.py \
    --out   output/asoca \
    --model output/mm-whs/full-stn/train/model \
    --test  data/config/inference.csv \
    --anatomix_checkpoints \
        saved_models/segmentation/seed_42/finetuned_MM-WHS0.1175.pth \
        saved_models/segmentation/seed_123/finetuned_MM-WHS0.1165.pth \
        saved_models/segmentation/seed_456/finetuned_MM-WHS0.1151.pth \
        saved_models/segmentation/seed_789/finetuned_MM-WHS0.1163.pth \
        saved_models/segmentation/seed_999/finetuned_MM-WHS0.1186.pth

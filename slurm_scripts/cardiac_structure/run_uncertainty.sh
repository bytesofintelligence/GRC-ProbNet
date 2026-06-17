#!/bin/bash

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=uncertainty_estimation

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u atlas-istn-anatomix.py \
    --out   output/mm-whs/full-stn \
    --model output/mm-whs/full-stn/train/model \
    --test  data/config/test.csv \
    --anatomix_checkpoints \
        saved_models/segmentation/seed_42/finetuned_MM-WHS0.1175.pth \
        saved_models/segmentation/seed_123/finetuned_MM-WHS0.1165.pth \
        saved_models/segmentation/seed_456/finetuned_MM-WHS0.1151.pth \
        saved_models/segmentation/seed_789/finetuned_MM-WHS0.1163.pth \
        saved_models/segmentation/seed_999/finetuned_MM-WHS0.1186.pth

#!/bin/bash
# Submit all 5 folds × 5 seeds = 25 ASOCA segmentation fine-tuning jobs.
#
# Usage:
#   bash run_coronary_seg_finetune.sh
#
# Each job trains an independently initialised Anatomix model for one
# (fold, seed) pair. Seeds differ only in random initialisation and
# augmentation randomness; the train/test split is identical per fold.

SEEDS=(42 123 456 789 999)

for FOLD in 1 2 3 4 5; do
    for SEED in "${SEEDS[@]}"; do
        sbatch \
            --partition=gpus24 \
            --gres=gpu:1 \
            --job-name=coronary_seg_ft_fold${FOLD}_seed${SEED} \
            --output=logs/experiment_4/anatomix_coronary_seg_finetune_fold${FOLD}_seed${SEED}_%j.log \
            --wrap="bash -c \"
                cd /vol/biomedic2/bglocker_studproj/<USERNAME> &&
                source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet &&
                cd grc-net &&
                python -u coronary-seg-finetune.py --fold ${FOLD} --seed ${SEED}
            \""
        echo "Submitted fold ${FOLD} seed ${SEED}"
    done
done

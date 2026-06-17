#!/bin/bash
# Run the coronary Atlas-ISTN pipeline for all 5 CV folds in parallel.
# Usage: sbatch run_coronary_atlas_istn.sh
# Each array task handles one fold; logs go to logs/experiment_4/coronary_atlas_istn_<fold>_<job>.log

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=coronary_atlas_istn
#SBATCH --output=logs/experiment_4/coronary_atlas_istn_fold_%a_%j.log

FOLD=fold_${SLURM_ARRAY_TASK_ID}

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u coronary-atlas-istn.py \
    --fold        "${FOLD}" \
    --config      data/config/asoca/config.json \
    --train       data/config/asoca/${FOLD}/train.csv \
    --val         data/config/asoca/${FOLD}/test.csv \
    --test        data/config/asoca/${FOLD}/test.csv \
    --out         output/asoca-coronary/${FOLD}/full-stn \
    --model       output/asoca-coronary/${FOLD}/full-stn/train/model \
    --stn         f \
    --dev         0

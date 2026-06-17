#!/bin/bash
# Run coronary classification for all 5 CV folds in parallel.
# Usage: sbatch run_coronary_classification.sh
# Each array task handles one fold; logs go to logs/experiment_4/coronary_classification_<fold>_<job>.log

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=coronary_classification
#SBATCH --output=logs/experiment_4/coronary_classification_fold_%a_%j.log

FOLD=fold_${SLURM_ARRAY_TASK_ID}

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

python -u coronary-classification.py --fold "${FOLD}"

#!/bin/bash
# Run multi-seed STN uncertainty inference for all 5 CV folds in parallel.
#
# Each array task handles one fold and runs two sequential passes:
#   Pass A — test split  (8 patients)  → output/asoca-coronary/fold_k/full-stn/uncertainty/seed_X/
#   Pass B — train split (32 patients) → output/asoca-coronary/fold_k/full-stn-train-unc/uncertainty/seed_X/
#
# Both passes use the SAME frozen trained STN from:
#   output/asoca-coronary/fold_k/full-stn/train/model/stn.pt
#
# The two separate output trees mirror how coronary-classification.py iterates:
#   train dataloader (sample_idx 0-31) → matched by Pass B sample_id 0-31
#   test  dataloader (sample_idx 0-7)  → matched by Pass A sample_id 0-7
#
# Prerequisites (both must be complete before submitting):
#   run_coronary_seg_finetune.sh  — 25 segmentation checkpoints
#   run_coronary_atlas_istn.sh    — 5 trained fold-specific STNs
#
# Usage:
#   sbatch run_coronary_uncertainty_seeds.sh

#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --job-name=coronary_unc_seeds
#SBATCH --output=logs/experiment_4/coronary_unc_seeds_fold_%a_%j.log

FOLD=fold_${SLURM_ARRAY_TASK_ID}
SEEDS=(42 123 456 789 999)

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

echo "========================================"
echo "Fold : ${FOLD}"
echo "Date : $(date)"
echo "Host : $(hostname)"
echo "========================================"

# ── Discover best-loss Anatomix checkpoint for each seed ─────────────────────
# Mirrors _find_best_asoca_fold_checkpoint in coronary-atlas-istn.py, applied
# per seed subdirectory.  Selects the checkpoint whose filename encodes the
# lowest validation loss (e.g. finetuned_asoca_fold_1_0.2215.pth → 0.2215).
# Falls back to the initial anatomix_trained checkpoint if no finetuned files exist.

CKPTS=()
echo ""
echo "--- Checkpoint discovery ---"
for SEED in "${SEEDS[@]}"; do
    CKPT=$(python3 -c "
import glob
seed_dir = 'saved_models/segmentation_asoca/${FOLD}/seed_${SEED}'
files = glob.glob(seed_dir + '/finetuned_asoca_${FOLD}_*.pth')
if not files:
    print(seed_dir + '/anatomix_trained_asoca_${FOLD}.pth')
else:
    print(min(files, key=lambda f: float(f.rsplit('_', 1)[-1][:-4])))
")
    echo "  seed_${SEED}: ${CKPT}"
    CKPTS+=("${CKPT}")
done

# ── Pass A: test split ────────────────────────────────────────────────────────
# Processes the 8 held-out test patients for this fold.
# sample_id 0-7 in the output NIfTIs matches sample_idx 0-7 from the test
# dataloader in coronary-classification.py.
#
# Output: output/asoca-coronary/${FOLD}/full-stn/uncertainty/seed_X/sample_N_*.nii.gz

echo ""
echo "--- Pass A: test split (8 patients) ---"
echo "    CSV   : data/config/asoca/${FOLD}/test.csv"
echo "    Output: output/asoca-coronary/${FOLD}/full-stn/uncertainty/"
echo "    Model : output/asoca-coronary/${FOLD}/full-stn/train/model/stn.pt"
echo ""

python -u coronary-atlas-istn.py \
    --fold  "${FOLD}" \
    --test  "data/config/asoca/${FOLD}/test.csv" \
    --out   "output/asoca-coronary/${FOLD}/full-stn" \
    --model "output/asoca-coronary/${FOLD}/full-stn/train/model" \
    --dev   0 \
    --anatomix_checkpoints "${CKPTS[@]}"

echo ""
echo "Pass A complete."

# ── Pass B: train split ───────────────────────────────────────────────────────
# Processes the 32 training patients for this fold.
# sample_id 0-31 in the output NIfTIs matches sample_idx 0-31 from the train
# dataloader in coronary-classification.py.
#
# --out is the only argument that differs from Pass A; --model still resolves
# to the same trained STN (default derivation is from --fold, not --out).
#
# Output: output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty/seed_X/sample_N_*.nii.gz

echo ""
echo "--- Pass B: train split (32 patients) ---"
echo "    CSV   : data/config/asoca/${FOLD}/train.csv"
echo "    Output: output/asoca-coronary/${FOLD}/full-stn-train-unc/uncertainty/"
echo "    Model : output/asoca-coronary/${FOLD}/full-stn/train/model/stn.pt  (same frozen STN)"
echo ""

python -u coronary-atlas-istn.py \
    --fold  "${FOLD}" \
    --test  "data/config/asoca/${FOLD}/train.csv" \
    --out   "output/asoca-coronary/${FOLD}/full-stn-train-unc" \
    --model "output/asoca-coronary/${FOLD}/full-stn/train/model" \
    --dev   0 \
    --anatomix_checkpoints "${CKPTS[@]}"

echo ""
echo "Pass B complete."
echo ""
echo "========================================"
echo "Fold ${FOLD} — both passes complete."
echo "Date : $(date)"
echo "========================================"

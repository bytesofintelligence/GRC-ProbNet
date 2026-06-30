#!/bin/bash

#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=mmwhs_unc_all

# Compute all uncertainty variants from MM-WHS multi-seed STN outputs.
#
# Reads from:  output/mm-whs/full-stn/uncertainty/seed_{42,123,456,789,999}/sample_N_*.nii.gz
# Writes to:   output/mm-whs/full-stn/uncertainty_analysis_<variant>/
#
# Variants run in order:
#   1. entropy_only_one  — single global predictive entropy per image
#   2. kl_only_one       — single global JSD value per image
#   3. kl                — per-structure JSD (7 features per image)
#   4. aleatoric_only_one — single global aleatoric uncertainty per image
#   5. aleatoric          — per-structure aleatoric uncertainty (7 features per image)

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

UNCERTAINTY_ROOT=output/mm-whs/full-stn/uncertainty
TEST_DIR=output/mm-whs/full-stn/test
OUT_BASE=output/mm-whs/full-stn

# echo "=== [1/5] entropy_only_one ==="
# python -u "cardiac_structure_files/compute_uncertainty-entropy_only_one.py" \
#     --uncertainty_root "$UNCERTAINTY_ROOT" \
#     --test_dir         "$TEST_DIR" \
#     --out_dir          "$OUT_BASE/uncertainty_analysis_entropy_only_one"

# echo "=== [2/5] kl_only_one ==="
# python -u cardiac_structure_files/compute_uncertainty_kl_only_one.py \
#     --uncertainty_root "$UNCERTAINTY_ROOT" \
#     --test_dir         "$TEST_DIR" \
#     --out_dir          "$OUT_BASE/uncertainty_analysis_kl_only_one"

echo "=== [3/5] kl (per-structure) ==="
python -u cardiac_structure_files/compute_uncertainty_kl.py \
    --uncertainty_root "$UNCERTAINTY_ROOT" \
    --test_dir         "$TEST_DIR" \
    --out_dir          "$OUT_BASE/uncertainty_analysis_kl"

# echo "=== [4/5] aleatoric_only_one ==="
# python -u cardiac_structure_files/compute_uncertainty_aleatoric_only_one.py \
#     --uncertainty_root "$UNCERTAINTY_ROOT" \
#     --test_dir         "$TEST_DIR" \
#     --out_dir          "$OUT_BASE/uncertainty_analysis_aleatoric_only_one"

echo "=== [5/5] aleatoric (per-structure) ==="
python -u cardiac_structure_files/compute_uncertainty_aleatoric.py \
    --uncertainty_root "$UNCERTAINTY_ROOT" \
    --test_dir         "$TEST_DIR" \
    --out_dir          "$OUT_BASE/uncertainty_analysis_aleatoric"

echo "=== All done ==="

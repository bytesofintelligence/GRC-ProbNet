#!/bin/bash
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --output=slurm.%N.%j.log
#SBATCH --job-name=grcnet_original_rerun
#SBATCH --time=24:00:00

set -e

cd /vol/biomedic2/bglocker_studproj/<USERNAME>
source miniconda3/bin/activate /vol/biomedic2/bglocker_studproj/<USERNAME>/miniconda3/envs/grcnet
cd grc-net

# ── Run identifiers ───────────────────────────────────────────────────────────
RUN_TAG=$(date +%Y%m%d_%H%M%S)
SEED=12345

SEG_DIR="saved_models/segmentation/seed_${SEED}"
ISTN_OUT="output/mm-whs/full-stn-rerun-${RUN_TAG}"
ISTN_MODEL="${ISTN_OUT}/train/model"
RESULTS_CSV="results_baseline_rerun_${RUN_TAG}.csv"

echo "============================================================"
echo "  GRC-Net original baseline pipeline re-run: ${RUN_TAG}"
echo "  Anatomix seed   : ${SEED}"
echo "  Anatomix dir    : ${SEG_DIR}"
echo "  STN output dir  : ${ISTN_OUT}"
echo "  Results CSV     : ${RESULTS_CSV}"
echo "============================================================"

# ── Stage 1: Anatomix fine-tuning ────────────────────────────────────────────
echo ""
echo "=== Stage 1: Anatomix fine-tuning (seed=${SEED}) ==="

# Re-run if no finetuned checkpoints exist in the seed directory yet
EXISTING_CKPT=$(ls "${SEG_DIR}"/finetuned_MM-WHS*.pth 2>/dev/null | head -1)
if [ -n "${EXISTING_CKPT}" ]; then
    echo "  Skipping — checkpoints already exist in ${SEG_DIR}."
else
    python -u cardiac_structure_files/anatomix-fine-tuning.py --seed ${SEED}
    echo "  Stage 1 complete."
fi

# Pick the best checkpoint = finetuned_MM-WHS*.pth with the lowest val loss
SEG_CKPT=$(python -c "
import glob, os
files = glob.glob('${SEG_DIR}/finetuned_MM-WHS*.pth')
best = min(files, key=lambda f: float(os.path.basename(f)[len('finetuned_MM-WHS'):-len('.pth')]))
print(best)
")
echo "  Best anatomix checkpoint: ${SEG_CKPT}"

# ── Stage 2: Atlas-ISTN training ─────────────────────────────────────────────
echo ""
echo "=== Stage 2: Atlas-ISTN training → ${ISTN_OUT} ==="

python -u cardiac_structure_files/atlas-istn-anatomix.py \
    --out   "${ISTN_OUT}" \
    --model "${ISTN_MODEL}" \
    --anatomix-model "${SEG_CKPT}"

echo "  Stage 2 complete → ${ISTN_MODEL}/stn.pt  (anatomix: ${SEG_CKPT})"

# ── Stage 3: Baseline classification ─────────────────────────────────────────
echo ""
echo "=== Stage 3: Baseline classification ==="

# Create a temp copy of the classification script with the new STN and atlas
# paths injected so the existing output/mm-whs/full-stn outputs are not touched.
TMP_CLASSIF="cardiac_structure_files/_tmp_classification_${RUN_TAG}.py"

sed \
    -e "s|stn_path = \"output/mm-whs/full-stn/train/model/stn.pt\"|stn_path = \"${ISTN_MODEL}/stn.pt\"|g" \
    -e "s|sitk.ReadImage(\"output/mm-whs/full-stn/train/model/atlas_labelmap_final.nii.gz\")|sitk.ReadImage(\"${ISTN_MODEL}/atlas_labelmap_final.nii.gz\")|g" \
    -e "s|_results_csv = f\"results_{_exp_tag}.csv\"|_results_csv = \"${RESULTS_CSV}\"|g" \
    cardiac_structure_files/Geo-Radio-Classification.py > "${TMP_CLASSIF}"

python -u "${TMP_CLASSIF}" \
    --anatomix-model "${SEG_CKPT}" \
    --uncertainty-csv output/asoca/uncertainty_analysis/metrics/per_structure_uncertainty.csv

CLASSIF_EXIT=$?
rm -f "${TMP_CLASSIF}"
echo "  Stage 3 complete (exit ${CLASSIF_EXIT})"

echo ""
echo "============================================================"
echo "  Re-run ${RUN_TAG} finished."
echo "  Best anatomix  : ${SEG_CKPT}"
echo "  STN + atlas    : ${ISTN_MODEL}/"
echo "  Results CSV    : ${RESULTS_CSV}"
echo "============================================================"

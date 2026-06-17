#!/bin/bash

# Master submission script — chains all 4 experiment jobs in order.
#
# Dependency chain:
#
#   Job 1: run_asoca_uncertainty_seeds.sh
#     └─ Job 2: run_compute_uncertainty_asoca.sh   (afterok Job 1)
#          └─ Job 3: run_classification_baseline.sh  (afterok Job 2)
#               └─ Job 4: run_classification_uncertainty.sh  (afterok Job 3)
#
# Jobs 3 and 4 run sequentially (not in parallel) to avoid competing for
# the same GPU while running 500 Optuna trials each.
#
# Usage:
#   bash submit_experiment_pipeline.sh
#
# To check status:
#   squeue -u $USER

set -e   # abort if any sbatch fails

cd /vol/biomedic2/bglocker_studproj/<USERNAME>/grc-net

JOB1=$(sbatch --parsable run_asoca_uncertainty_seeds.sh)
echo "Submitted Job 1 (asoca_unc_seeds)       : $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 run_compute_uncertainty_asoca.sh)
echo "Submitted Job 2 (asoca_unc_compute)     : $JOB2  [after $JOB1]"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 run_classification_baseline.sh)
echo "Submitted Job 3 (classify_baseline)     : $JOB3  [after $JOB2]"

JOB4=$(sbatch --parsable --dependency=afterok:$JOB2 run_classification_uncertainty.sh)
echo "Submitted Job 4 (classify_uncertainty)  : $JOB4  [after $JOB2]"

echo ""
echo "All jobs queued. Pipeline will produce:"
echo "  output/asoca/uncertainty/seed_*/sample_N_*.nii.gz"
echo "  output/asoca/uncertainty_analysis/metrics/per_structure_uncertainty.csv"
echo "  results_baseline.csv"
echo "  results_uncertainty.csv"
echo ""
echo "Monitor with:  squeue -u $USER"
echo "Logs:          slurm.<node>.<jobid>.log"

#!/bin/bash
#SBATCH --job-name=evaluate_rag
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/rag_project/logs/evaluate_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/rag_project/logs/evaluate_%j.err
#SBATCH --environment=indexing_pipeline

# Stop on errors
set -eo pipefail

# --- Argument Validation ---
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch evaluate.sbatch <excel_path> <top_k>"
    echo "Example: sbatch evaluate.sbatch /path/to/questions.xlsx 100"
    exit 1
fi

EXCEL_PATH="$1"
TOP_K="$2"

echo "[sbatch] running on $(hostname)"
echo "[sbatch] SLURM_JOB_ID: $SLURM_JOB_ID"
echo "[sbatch] Excel path: $EXCEL_PATH"
echo "[sbatch] Top-K: $TOP_K"

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

echo "[sbatch] Starting evaluation..."

# Run evaluation
python3 evaluate_rag.py \
    --excel "$EXCEL_PATH" \
    --top-k "$TOP_K"

echo "[sbatch] Evaluation finished successfully"

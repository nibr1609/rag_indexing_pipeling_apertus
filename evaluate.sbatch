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
if [ -z "$1" ]; then
    echo "Usage: sbatch evaluate.sbatch <excel_path> [options]"
    echo "Example: sbatch evaluate.sbatch /path/to/questions.xlsx"
    echo "Example: sbatch evaluate.sbatch /path/to/questions.xlsx --all-scenarios"
    echo "Example: sbatch evaluate.sbatch /path/to/questions.xlsx --use-reranker"
    echo "Note: Always retrieves k=100 results and computes accuracy at k=1,3,5,10,25,50,100"
    echo ""
    echo "Options:"
    echo "  --all-scenarios: Run all 4 evaluation scenarios"
    echo "  --use-query-expansion: Enable query expansion only"
    echo "  --use-reranker: Enable reranker only"
    echo "  --compare-query-expansion: (Deprecated) Use --all-scenarios instead"
    exit 1
fi

EXCEL_PATH="$1"
EXTRA_ARGS="${@:2}"  # Capture all arguments after the first one

echo "[sbatch] running on $(hostname)"
echo "[sbatch] SLURM_JOB_ID: $SLURM_JOB_ID"
echo "[sbatch] Excel path: $EXCEL_PATH"
echo "[sbatch] Top-K: 100 (fixed)"
echo "[sbatch] Extra args: $EXTRA_ARGS"

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

echo "[sbatch] Starting evaluation..."

# Run evaluation with k=100 (always retrieves 100 results, computes accuracy at multiple k values)
python3 evaluate_rag.py \
    --excel "$EXCEL_PATH" \
    --top-k 100 \
    $EXTRA_ARGS

echo "[sbatch] Evaluation finished successfully"

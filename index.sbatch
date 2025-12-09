#!/bin/bash
#SBATCH --job-name=index_webarchive
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/rag_project/logs/output_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/rag_project/logs/output_%j.err
#SBATCH --environment=indexing_pipeline 

# Stop the script if a command fails
set -eo pipefail

# --- Argument Validation ---

# Validate if both arguments are provided ($1 and $2)
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch run_pipeline.sbatch <warc_dir> <excel_path>"
    exit 1
fi

WARC_INPUT_DIR="$1"
EXCEL_PATH="$2" # This variable now holds the path to your Excel file

echo "[sbatch] running on $(hostname)"
echo "[sbatch] SLURM_JOB_ID: $SLURM_JOB_ID"
echo "[sbatch] Processing WARC directory: $WARC_INPUT_DIR"
echo "[sbatch] Using filter Excel path: $EXCEL_PATH"

PROJECT_DIR="$SLURM_SUBMIT_DIR" 
cd "$PROJECT_DIR"

echo "[sbatch] Starting Python pipeline..."

# --- Corrected Python Function Call ---
# Pass both arguments to the Python script with their required flags

python run_indexing_pipeline.py \
    --warc-input-dir "$WARC_INPUT_DIR" \
    --topics-excel-path "$EXCEL_PATH"

echo "[sbatch] Pipeline finished successfully"
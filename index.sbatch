#!/bin/bash
#SBATCH --job-name=index_webarchive
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/rag_project/logs/output_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/rag_project/logs/output_%j.err
#SBATCH --environment=indexing_pipeline

# Stop the script if a command fails
set -eo pipefail

# --- Argument Validation ---

# Validate if both required arguments are provided ($1 and $2)
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch index.sbatch <warc_dir> <excel_path> [indexed_files_path]"
    echo "  warc_dir: Path to WARC files directory"
    echo "  excel_path: Path to topics Excel file"
    echo "  indexed_files_path: (Optional) Path to track already indexed files"
    exit 1
fi

WARC_INPUT_DIR="$1"
EXCEL_PATH="$2"
INDEXED_FILES_PATH="$3"  # Optional third argument

echo "[sbatch] running on $(hostname)"
echo "[sbatch] SLURM_JOB_ID: $SLURM_JOB_ID"
echo "[sbatch] Processing WARC directory: $WARC_INPUT_DIR"
echo "[sbatch] Using filter Excel path: $EXCEL_PATH"

if [ -n "$INDEXED_FILES_PATH" ]; then
    echo "[sbatch] Using indexed files tracking: $INDEXED_FILES_PATH"
fi

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

echo "[sbatch] Starting Python pipeline..."

# --- Python Function Call ---
# Build command with optional indexed files path

if [ -n "$INDEXED_FILES_PATH" ]; then
    python run_indexing_pipeline.py \
        --warc-input-dir "$WARC_INPUT_DIR" \
        --topics-excel-path "$EXCEL_PATH" \
        --indexed-files-path "$INDEXED_FILES_PATH"
else
    python run_indexing_pipeline.py \
        --warc-input-dir "$WARC_INPUT_DIR" \
        --topics-excel-path "$EXCEL_PATH"
fi

echo "[sbatch] Pipeline finished successfully"
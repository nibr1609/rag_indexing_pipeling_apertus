#!/bin/bash
#SBATCH --job-name=query_webarchive
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/rag_project/logs/query_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/rag_project/logs/query_%j.err
#SBATCH --environment=indexing_pipeline

# Stop the script if a command fails
set -eo pipefail

# --- Argument Validation ---
if [ -z "$1" ]; then
    echo "Usage: sbatch query.sbatch \"<your query>\" [top_k]"
    echo "Example: sbatch query.sbatch \"What is machine learning?\" 5"
    exit 1
fi

QUERY="$1"
TOP_K="${2:-5}"  # Default to 5 if not provided

echo "[sbatch] running on $(hostname)"
echo "[sbatch] SLURM_JOB_ID: $SLURM_JOB_ID"
echo "[sbatch] Query: $QUERY"
echo "[sbatch] Top K: $TOP_K"

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

echo "[sbatch] Starting query..."

# Run the query
python3 -c "
from query_elasticsearch import simple_search, print_search_results
import os
from dotenv import load_dotenv

load_dotenv()

print('Executing semantic search...')
print()

results = simple_search(
    query='${QUERY}',
    index_name=os.getenv('INDEX_NAME', 'ethz_archive_index'),
    es_url=os.getenv('ES_URL', 'https://es.swissai.cscs.ch'),
    top_k=${TOP_K},
    es_user=os.getenv('ELASTIC_USERNAME'),
    es_password=os.getenv('ELASTIC_PASSWORD')
)

print_search_results(results)
"

echo "[sbatch] Query finished successfully"

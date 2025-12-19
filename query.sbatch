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
    echo "Usage: sbatch query.sbatch \"<your query>\" [top_k] [--use-reranker] [--use-query-expansion]"
    echo "Example: sbatch query.sbatch \"What is machine learning?\" 5"
    echo "Example: sbatch query.sbatch \"What is machine learning?\" 10 --use-reranker"
    echo "Example: sbatch query.sbatch \"What is machine learning?\" 10 --use-query-expansion --use-reranker"
    exit 1
fi

QUERY="$1"
TOP_K="${2:-5}"  # Default to 5 if not provided

# Parse optional flags
USE_RERANKER=false
USE_QUERY_EXPANSION=false

for arg in "${@:3}"; do
    case "$arg" in
        --use-reranker)
            USE_RERANKER=true
            ;;
        --use-query-expansion)
            USE_QUERY_EXPANSION=true
            ;;
    esac
done

echo "[sbatch] running on $(hostname)"
echo "[sbatch] SLURM_JOB_ID: $SLURM_JOB_ID"
echo "[sbatch] Query: $QUERY"
echo "[sbatch] Top K: $TOP_K"
echo "[sbatch] Use Reranker: $USE_RERANKER"
echo "[sbatch] Use Query Expansion: $USE_QUERY_EXPANSION"

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

echo "[sbatch] Starting query..."

# Convert bash booleans to Python booleans
if [ "$USE_RERANKER" = "true" ]; then
    PYTHON_USE_RERANKER="True"
else
    PYTHON_USE_RERANKER="False"
fi

if [ "$USE_QUERY_EXPANSION" = "true" ]; then
    PYTHON_USE_QUERY_EXPANSION="True"
else
    PYTHON_USE_QUERY_EXPANSION="False"
fi

# Run the query
python3 -c "
from query_elasticsearch import simple_search, print_search_results
import os
from dotenv import load_dotenv

load_dotenv()

print('Executing semantic search...')
print('Use Reranker: ${USE_RERANKER}')
print('Use Query Expansion: ${USE_QUERY_EXPANSION}')
print()

results = simple_search(
    query='${QUERY}',
    index_name=os.getenv('INDEX_NAME', 'ethz_archive_index'),
    es_url=os.getenv('ES_URL', 'https://es.swissai.cscs.ch'),
    top_k=${TOP_K},
    es_user=os.getenv('ELASTIC_USERNAME'),
    es_password=os.getenv('ELASTIC_PASSWORD'),
    use_reranker=${PYTHON_USE_RERANKER},
    use_query_expansion=${PYTHON_USE_QUERY_EXPANSION}
)

print_search_results(results)
"

echo "[sbatch] Query finished successfully"

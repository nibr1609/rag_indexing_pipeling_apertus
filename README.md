# ETH Zurich Web Archive RAG System

This repository provides a complete pipeline for processing ETH Zurich web archive files (WARC format), indexing them into Elasticsearch, and evaluating retrieval performance with advanced features like query expansion and reranking.

**Note:** This guide focuses on running the system on the CSCS cluster using SLURM job submission.

## Features

- **Data Processing Pipeline**
  - Extract HTML and PDF files from WARC archives
  - Combine domain-specific HTML files by timestamp
  - Convert HTML to Markdown format

- **Indexing & Search**
  - Index documents to Elasticsearch with semantic embeddings
  - Semantic search with vector similarity
  - Query expansion for improved retrieval
  - Cross-encoder reranking for better result quality

- **Evaluation Framework**
  - Compare multiple retrieval strategies
  - Automated accuracy metrics at different k values
  - LaTeX table generation for results
  - Support for 4 evaluation scenarios:
    1. Baseline retrieval
    2. Query expansion + retrieval
    3. Retrieval + reranking
    4. Query expansion + retrieval + reranking

## Prerequisites

- Access to CSCS cluster (Alps/Daint)
- Access to Elasticsearch instance (https://es.swissai.cscs.ch)
- Access to embedding service
- (Optional) Access to OpenAI-compatible API for query expansion

## Setup on CSCS Cluster

### Step 1: Build the Container

The CSCS cluster requires a SquashFS container for running the pipeline. Build it from a compute node:

```bash
# Request an interactive compute node
srun --nodes=1 --time=01:00:00 --partition=normal --account=large-sc-2 --container-writable --pty bash

# Navigate to your project directory
cd /iopsstor/scratch/cscs/$USER/path/to/project/

# Build the container image
podman build -t ethz_cpu_rag:v1 .

# Convert to SquashFS format for enroot
enroot import -o ethz_cpu_rag.sqsh podman://ethz_cpu_rag:v1

# Exit the interactive session
exit
```

### Step 2: Create Enroot Configuration

Create the configuration file `~/.edf/rag_env.toml`:

```toml
image = "/iopsstor/scratch/cscs/<your_username>/path/to/project/ethz_cpu_rag.sqsh"
mounts = [
    "/iopsstor/scratch/cscs/<your_username>:/iopsstor/scratch/cscs/<your_username>"
]
writable = true
```

Replace `<your_username>` with your CSCS username.

### Step 3: Configure Environment Variables

Create a `.env` file in your project directory:

```bash
# Elasticsearch Configuration
ELASTIC_USERNAME=your_elasticsearch_username
ELASTIC_PASSWORD=your_elasticsearch_password
ES_URL=https://es.swissai.cscs.ch
INDEX_NAME=ethz_webarchive

# Embedding Service
EMBEDDING_SERVICE_URL=http://your-embedding-service-url

# Query Expansion (Optional)
QUERY_EXPANSION_API_KEY=your_openai_api_key
QUERY_EXPANSION_BASE_URL=https://api.swissai.cscs.ch/v1
QUERY_EXPANSION_MODEL=gpt-4
```

**Required variables:**
- `ELASTIC_USERNAME`: Your Elasticsearch username
- `ELASTIC_PASSWORD`: Your Elasticsearch password
- `EMBEDDING_SERVICE_URL`: URL for the embedding service

**Optional variables:**
- `QUERY_EXPANSION_API_KEY`: API key for query expansion (required if using `--use-query-expansion`)
- `QUERY_EXPANSION_BASE_URL`: Base URL for OpenAI-compatible API
- `QUERY_EXPANSION_MODEL`: Model to use for query expansion

### Step 4: Create Log Directory

```bash
mkdir -p /iopsstor/scratch/cscs/$USER/rag_project/logs
```

## Usage

All operations are submitted as SLURM jobs using the provided `.sbatch` scripts.

### 1. Indexing Pipeline

Process WARC files and index them to Elasticsearch:

```bash
sbatch index.sbatch
```

This will:
1. Extract HTML and PDF files from WARC archives
2. Combine HTML files by domain and timestamp
3. Convert HTML to Markdown
4. Generate embeddings using the remote embedding service
5. Index documents to Elasticsearch

The pipeline processes files from `./data/ethz_websites_2022-2025_examples/` and outputs to `./output/`.

**Monitor the job:**
```bash
# Check job status
squeue -u $USER

# View output logs
tail -f /iopsstor/scratch/cscs/$USER/rag_project/logs/index_<job_id>.out

# View error logs
tail -f /iopsstor/scratch/cscs/$USER/rag_project/logs/index_<job_id>.err
```

### 2. Querying Documents

Query the indexed documents using different strategies.

#### Basic Query

```bash
sbatch query.sbatch "What is machine learning?" 10
```

Arguments:
- First argument: Query string (in quotes)
- Second argument: Number of results to retrieve (default: 5)

#### Query with Reranking

```bash
sbatch query.sbatch "What is machine learning?" 20 --use-reranker
```

The reranker uses a cross-encoder model to improve result quality by rescoring the top-k results.

#### Query with Query Expansion

```bash
sbatch query.sbatch "What is machine learning?" 20 --use-query-expansion
```

Query expansion uses an LLM to reformulate the query for better retrieval.

#### Full Pipeline (Query Expansion + Reranking)

```bash
sbatch query.sbatch "What is machine learning?" 20 --use-query-expansion --use-reranker
```

This runs the complete pipeline: expand query → retrieve → rerank with original query.

**Monitor the job:**
```bash
# View output
tail -f /iopsstor/scratch/cscs/$USER/rag_project/logs/query_<job_id>.out
```

### 3. Evaluating the RAG System

The evaluation system compares different retrieval strategies using a ground truth dataset.

#### Prepare Evaluation Data

Create an Excel file (e.g., `questions.xlsx`) with columns:
- `question`: The query text
- `relevant_doc_1`: URL of the first relevant document
- `relevant_doc_2`: URL of the second relevant document (optional)

Example:
| question | relevant_doc_1 | relevant_doc_2 |
|----------|---------------|----------------|
| What is ETH Zurich's admission process? | https://ethz.ch/en/studies/registration-application.html | https://ethz.ch/en/studies/bachelor.html |

Place the file in your project directory or on the scratch filesystem.

#### Run All 4 Evaluation Scenarios (Recommended)

```bash
sbatch evaluate.sbatch /path/to/questions.xlsx --all-scenarios
```

This evaluates:
1. **Baseline retrieval** - Standard semantic search
2. **Query expansion + retrieval** - Expanded queries for better recall
3. **Retrieval + reranking** - Cross-encoder reranking of results
4. **Query expansion + retrieval + reranking** - Full pipeline (reranks with original query)

The evaluation retrieves k=100 results and computes accuracy at k=1, 3, 5, 10, 25, 50, 100.

#### Run Specific Scenarios

**Baseline only:**
```bash
sbatch evaluate.sbatch /path/to/questions.xlsx
```

**Query expansion only:**
```bash
sbatch evaluate.sbatch /path/to/questions.xlsx --use-query-expansion
```

**Reranker only:**
```bash
sbatch evaluate.sbatch /path/to/questions.xlsx --use-reranker
```

**Full pipeline:**
```bash
sbatch evaluate.sbatch /path/to/questions.xlsx --use-query-expansion --use-reranker
```

**Monitor the job:**
```bash
# Check job status (evaluation can take 1-2 hours)
squeue -u $USER

# View progress in real-time
tail -f /iopsstor/scratch/cscs/$USER/rag_project/logs/evaluate_<job_id>.out
```

#### Understanding Evaluation Results

The evaluation outputs:

1. **Console output** (in log files) - Progress and summary statistics
2. **JSON file** (`output/rag_evaluation/rag_evaluation_*.json`) - Detailed results for each question and scenario
3. **LaTeX table** (`output/evaluation/accuracy_table.tex`) - Formatted accuracy table for papers

Example output:
```
======================================================================
SCENARIO 1/4: Retrieval Only
======================================================================
[1/50] What is ETH Zurich's admission process?...
  ✓ SUCCESS
[2/50] How do I apply for a PhD program?...
  ✗ FAILURE
...

======================================================================
ACCURACY AT DIFFERENT K VALUES
======================================================================

Retrieval Only:
  Accuracy @ k=  1: 45.0%
  Accuracy @ k=  3: 62.0%
  Accuracy @ k=  5: 71.0%
  Accuracy @ k= 10: 80.0%
  ...

Query Exp. + Retrieval:
  Accuracy @ k=  1: 52.0%
  Accuracy @ k=  3: 68.0%
  ...
```

## Architecture

### Modular Components

1. **query_elasticsearch.py** - Core search functionality
   - `Reranker` class - Cross-encoder reranking
   - `simple_search()` - Main search function with optional query expansion and reranking

2. **query_expansion.py** - Query expansion using LLMs
   - Expands user queries for better retrieval
   - Configurable via environment variables

3. **evaluate_rag.py** - Evaluation framework
   - Compares multiple retrieval strategies
   - Generates accuracy metrics and LaTeX tables

### Query Flow

```
User Query
    ↓
[Query Expansion] (optional)
    ↓
Elasticsearch Retrieval
    ↓
[Reranking] (optional)
    ↓
Results
```

**Important:** In the full pipeline scenario (query expansion + reranking), the reranking step uses the **original query**, not the expanded query, for better semantic matching.

## SLURM Configuration

### Default Job Parameters

All sbatch scripts use these default settings:

```bash
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --environment=indexing_pipeline
```

**Time limits:**
- `index.sbatch`: 4 hours
- `query.sbatch`: 30 minutes
- `evaluate.sbatch`: 2 hours

### Customizing Job Resources

To modify resource allocation, edit the `#SBATCH` directives in the respective `.sbatch` files:

```bash
#SBATCH --time=04:00:00        # Increase time limit
#SBATCH --cpus-per-task=8      # Increase CPU cores
#SBATCH --mem=32G              # Add memory limit
```

## Project Structure

```
ethz_webarchive/
├── data/                           # Input WARC files
├── output/                         # Processing outputs
│   ├── html_raw/                  # Extracted HTML files
│   ├── pdf_raw/                   # Extracted PDF files
│   ├── html_combined/             # Combined HTML by domain
│   ├── markdown/                  # Converted Markdown files
│   ├── mappings/                  # Domain and timestamp mappings
│   └── rag_evaluation/            # Evaluation results
├── prep_warc_files.py             # WARC extraction
├── combine_domains.py             # Domain combination
├── html_combined_to_markdown.py   # HTML to Markdown conversion
├── index_to_elasticsearch.py      # Elasticsearch indexing
├── query_elasticsearch.py         # Search & reranking
├── query_expansion.py             # Query expansion
├── evaluate_rag.py                # Evaluation framework
├── remote_embedding.py            # Remote embedding client
├── run_indexing_pipeline.py       # Main pipeline script
├── index.sbatch                   # SLURM script for indexing
├── query.sbatch                   # SLURM script for queries
├── evaluate.sbatch                # SLURM script for evaluation
├── Dockerfile                     # Container definition
├── env.yml                        # Mamba/Conda environment
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## Troubleshooting

### Container Build Failures

If the container build fails, check:
```bash
# Verify podman is available
podman --version

# Check disk space
df -h /iopsstor/scratch/cscs/$USER
```

### Job Failures

Check the error logs:
```bash
cat /iopsstor/scratch/cscs/$USER/rag_project/logs/<job_type>_<job_id>.err
```

Common issues:
- **Missing .env file**: Ensure `.env` is in the project directory with correct credentials
- **Elasticsearch connection**: Verify `ES_URL`, `ELASTIC_USERNAME`, `ELASTIC_PASSWORD`
- **Embedding service timeout**: Check `EMBEDDING_SERVICE_URL` is accessible from compute nodes

### Slow Evaluation Performance

The system includes optimizations to prevent slowdowns:
- Automatic cleanup of Elasticsearch connections
- Thread limiting for OpenBLAS (set to 4 threads)

If evaluation is still slow, reduce the dataset size or retrieve fewer results:
```bash
# This modifies the evaluate.sbatch script to use --top-k 50
```

### Query Expansion Failures

If query expansion fails, the system automatically falls back to the original query. Check:
```bash
# Verify API key is set
grep QUERY_EXPANSION_API_KEY .env

# Test API connectivity (from compute node)
curl -H "Authorization: Bearer $QUERY_EXPANSION_API_KEY" $QUERY_EXPANSION_BASE_URL/models
```

### Reranker Errors

Ensure sentence-transformers is installed in the container. Rebuild if necessary:
```bash
# Check requirements.txt includes:
# sentence-transformers>=2.2.0
```

## Performance Tips

1. **Use query expansion** when queries are short or ambiguous - improves recall
2. **Use reranking** when you need high precision in top results
3. **Adjust top_k** based on your use case:
   - Higher values (100+) for better recall
   - Lower values (10-20) for faster execution
4. **Run evaluation with --all-scenarios** to compare all strategies in one job

## Monitoring Jobs

### Useful SLURM Commands

```bash
# List your running/pending jobs
squeue -u $USER

# Get detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>

# View accounting info for completed jobs
sacct -j <job_id> --format=JobID,JobName,Elapsed,State,ExitCode
```

### Log Files

All job outputs are saved to:
```
/iopsstor/scratch/cscs/$USER/rag_project/logs/
```

File naming convention:
- `index_<job_id>.out` / `index_<job_id>.err`
- `query_<job_id>.out` / `query_<job_id>.err`
- `evaluate_<job_id>.out` / `evaluate_<job_id>.err`

## License

This project is part of ETH Zurich's web archiving efforts.

## Credit

Inspired by [this repo](https://github.com/rashitig/ethz_webarchive)

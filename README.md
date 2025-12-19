# ETH Zurich Web Archive RAG System

This repository provides a complete pipeline for processing ETH Zurich web archive files (WARC format), indexing them into Elasticsearch, and evaluating retrieval performance with advanced features like query expansion and reranking.

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

- [Mamba](https://mamba.readthedocs.io/) or [Conda](https://docs.conda.io/)
- Access to an Elasticsearch instance
- Access to an embedding service (or local Ollama)
- (Optional) Access to OpenAI-compatible API for query expansion

## Installation

### 1. Clone this repository

### 2. Create the environment

Using mamba (recommended for faster installation):

```bash
mamba env create -f env.yml
mamba activate rag
```

Or using conda:

```bash
conda env create -f env.yml
conda activate rag
```

### 3. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and configure your credentials:

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

## Usage

### 1. Indexing Pipeline

To process WARC files and index them to Elasticsearch:

```bash
python run_indexing_pipeline.py
```

Or on CSCS cluster:

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

### 2. Querying Documents

#### Basic Query (Local)

```bash
python -c "
from query_elasticsearch import simple_search, print_search_results
import os
from dotenv import load_dotenv

load_dotenv()

results = simple_search(
    query='What is machine learning?',
    index_name=os.getenv('INDEX_NAME'),
    es_url=os.getenv('ES_URL'),
    top_k=5,
    es_user=os.getenv('ELASTIC_USERNAME'),
    es_password=os.getenv('ELASTIC_PASSWORD')
)

print_search_results(results)
"
```

#### Query with Options

**With Query Expansion:**

```bash
python -c "
from query_elasticsearch import simple_search, print_search_results
import os
from dotenv import load_dotenv

load_dotenv()

results = simple_search(
    query='What is machine learning?',
    index_name=os.getenv('INDEX_NAME'),
    es_url=os.getenv('ES_URL'),
    top_k=10,
    es_user=os.getenv('ELASTIC_USERNAME'),
    es_password=os.getenv('ELASTIC_PASSWORD'),
    use_query_expansion=True
)

print_search_results(results)
"
```

**With Reranking:**

```bash
python -c "
from query_elasticsearch import simple_search, print_search_results
import os
from dotenv import load_dotenv

load_dotenv()

results = simple_search(
    query='What is machine learning?',
    index_name=os.getenv('INDEX_NAME'),
    es_url=os.getenv('ES_URL'),
    top_k=20,
    es_user=os.getenv('ELASTIC_USERNAME'),
    es_password=os.getenv('ELASTIC_PASSWORD'),
    use_reranker=True
)

print_search_results(results)
"
```

**Full Pipeline (Query Expansion + Reranking):**

```bash
python -c "
from query_elasticsearch import simple_search, print_search_results
import os
from dotenv import load_dotenv

load_dotenv()

results = simple_search(
    query='What is machine learning?',
    index_name=os.getenv('INDEX_NAME'),
    es_url=os.getenv('ES_URL'),
    top_k=20,
    es_user=os.getenv('ELASTIC_USERNAME'),
    es_password=os.getenv('ELASTIC_PASSWORD'),
    use_query_expansion=True,
    use_reranker=True
)

print_search_results(results)
"
```

#### Query on CSCS Cluster

**Basic query:**

```bash
sbatch query.sbatch "What is machine learning?" 10
```

**With reranker:**

```bash
sbatch query.sbatch "What is machine learning?" 20 --use-reranker
```

**With query expansion and reranker:**

```bash
sbatch query.sbatch "What is machine learning?" 20 --use-query-expansion --use-reranker
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

#### Run Evaluation

**Run all 4 scenarios (recommended):**

```bash
python evaluate_rag.py --excel questions.xlsx --all-scenarios
```

Or on CSCS cluster:

```bash
sbatch evaluate.sbatch questions.xlsx --all-scenarios
```

This evaluates:

1. **Baseline retrieval** - Standard semantic search
2. **Query expansion + retrieval** - Expanded queries for better recall
3. **Retrieval + reranking** - Cross-encoder reranking of results
4. **Query expansion + retrieval + reranking** - Full pipeline

**Run specific scenario:**

Baseline only:

```bash
python evaluate_rag.py --excel questions.xlsx
```

Query expansion only:

```bash
python evaluate_rag.py --excel questions.xlsx --use-query-expansion
```

Reranker only:

```bash
python evaluate_rag.py --excel questions.xlsx --use-reranker
```

Full pipeline:

```bash
python evaluate_rag.py --excel questions.xlsx --use-query-expansion --use-reranker
```

#### Evaluation Options

```bash
python evaluate_rag.py \
  --excel path/to/questions.xlsx \
  --top-k 100 \                    # Number of results to retrieve (default: 100)
  --rerank-top-k 100 \             # Number to keep after reranking (default: 100)
  --all-scenarios \                # Run all 4 evaluation scenarios
  --output results.json \          # Custom output file (optional)
  --latex-output table.tex         # Custom LaTeX output (optional)
```

#### Understanding Evaluation Results

The evaluation outputs:

1. **Console output** - Progress and summary statistics
2. **JSON file** - Detailed results for each question and scenario
3. **LaTeX table** - Formatted accuracy table for papers

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

The system is designed with modularity in mind:

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

## Running on CSCS Cluster

### Container Setup

The CSCS cluster requires a SquashFS container. Follow these steps:

#### Step 1: Build the Container

From a compute node:

```bash
srun --nodes=1 --time=01:00:00 --partition=normal --account=large-sc-2 --container-writable --pty bash
cd /iopsstor/scratch/cscs/$USER/path/to/project/
podman build -t ethz_cpu_rag:v1 .
enroot import -o ethz_cpu_rag.sqsh podman://ethz_cpu_rag:v1
exit
```

#### Step 2: Create Enroot Configuration

Create `~/.edf/rag_env.toml`:

```toml
image = "/iopsstor/scratch/cscs/<your_username>/path/to/project/ethz_cpu_rag.sqsh"
mounts = [
    "/iopsstor/scratch/cscs/<your_username>:/iopsstor/scratch/cscs/<your_username>"
]
writable = true
```

#### Step 3: Submit Jobs

```bash
# Indexing
sbatch index.sbatch

# Querying
sbatch query.sbatch "your query" 10 --use-reranker

# Evaluation
sbatch evaluate.sbatch /path/to/questions.xlsx --all-scenarios
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
├── env.yml                        # Mamba/Conda environment
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## Advanced Usage

### Custom Reranking Query

You can specify a different query for reranking:

```python
from query_elasticsearch import simple_search

results = simple_search(
    query="expanded query for retrieval",
    use_query_expansion=False,
    use_reranker=True,
    rerank_query="original user question",  # Use this for reranking
    rerank_top_k=50
)
```

### Programmatic Evaluation

```python
from evaluate_rag import evaluate_question, load_questions_from_excel
import os
from dotenv import load_dotenv

load_dotenv()

es_config = {
    'index_name': os.getenv('INDEX_NAME'),
    'es_url': os.getenv('ES_URL'),
    'es_user': os.getenv('ELASTIC_USERNAME'),
    'es_password': os.getenv('ELASTIC_PASSWORD')
}

questions_data = load_questions_from_excel('questions.xlsx')

for question, relevant_docs in questions_data:
    result = evaluate_question(
        question=question,
        relevant_docs=relevant_docs,
        es_config=es_config,
        top_k=100,
        use_query_expansion=True,
        use_reranker=True,
        rerank_query=question,  # Use original for reranking
        rerank_top_k=100
    )
    print(f"Question: {question}")
    print(f"Success: {result['success']}")
    print(f"Found {len(result['found_docs'])}/{len(relevant_docs)} docs")
```

### Embedding service issues

Verify the embedding service is accessible:

```bash
curl $EMBEDDING_SERVICE_URL/health
```

### Query expansion failures

Ensure your API key is set correctly:

```bash
echo $QUERY_EXPANSION_API_KEY
```

If query expansion fails, the system will automatically fall back to the original query.

### Reranker not available

Install sentence-transformers:

```bash
pip install sentence-transformers
```

### Memory issues during evaluation

Reduce batch sizes or top_k values:

```bash
python evaluate_rag.py --excel questions.xlsx --top-k 50 --rerank-top-k 25
```

## Performance Tips

1. **Use query expansion** when queries are short or ambiguous
2. **Use reranking** when you need high precision in top results
3. **Adjust top_k** based on your use case (higher for recall, lower for speed)
4. **Run evaluations on cluster** for faster processing with multiple questions

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is part of ETH Zurich's web archiving efforts.

## Credit

Inspired by [this repo](https://github.com/rashitig/ethz_webarchive)

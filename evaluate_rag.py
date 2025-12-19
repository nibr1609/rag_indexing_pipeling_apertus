#!/usr/bin/env python3
"""
RAG System Evaluation Script

Evaluates the RAG system by:
1. Reading questions and relevant docs from Excel
2. Querying the system for each question
3. Optionally reranking results with a cross-encoder
4. Checking if relevant docs are in top-K results
5. Computing accuracy metrics

Usage:
    python evaluate_rag.py --excel path/to/evaluate.xlsx --top-k 100
    python evaluate_rag.py --excel questions.xlsx --use-reranker --rerank-top-k 100
    python evaluate_rag.py --excel questions.xlsx --use-query-expansion --use-reranker
"""

import os
import sys

# Set OpenBLAS threads to avoid warnings and resource contention
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import argparse
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import urlparse, urlunparse
from query_elasticsearch import simple_search, RERANKER_AVAILABLE
from datetime import datetime
import json
import time

# Load environment
load_dotenv()


# ----------------------------
# URL NORMALIZATION
# ----------------------------
def normalize_url(url):
    """
    Normalize URL for robust matching.
    Handles:
    - Removes fragments (#...)
    - Removes trailing slashes
    - Removes index.html/index.htm
    - Strips file extensions (.html, .htm, .pdf, .md) for suffix-invariant matching
    - Lowercases
    - Removes query parameters
    
    Examples:
        https://ethz.ch/page.pdf -> https://ethz.ch/page
        https://ethz.ch/page.html -> https://ethz.ch/page
        https://ethz.ch/services/ -> https://ethz.ch/services
        https://ethz.ch/index.html -> https://ethz.ch
    """
    if not url or not isinstance(url, str):
        return None
    
    url = url.strip()
    if not url:
        return None
    
    # Parse URL
    parsed = urlparse(url)
    
    # Remove fragment (#...) and query parameters (?..)
    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        '',  # params
        '',  # query - remove if you want to ignore ?param=value
        ''   # fragment
    ))
    
    # Lowercase
    normalized = normalized.lower()
    
    # Remove trailing slash
    if normalized.endswith('/'):
        normalized = normalized[:-1]
    
    # Remove /index.html or /index.htm at the end
    if normalized.endswith('/index.html'):
        normalized = normalized[:-11]
    elif normalized.endswith('/index.htm'):
        normalized = normalized[:-10]
    
    # Strip file extensions for suffix-invariant matching
    for ext in ['.html', '.htm', '.pdf', '.md']:
        if normalized.endswith(ext):
            normalized = normalized[:-len(ext)]
            break
    
    return normalized


def is_url_match(result_url, expected_url):
    """
    Check if result URL matches expected URL using normalized comparison.
    
    Args:
        result_url: URL from search results
        expected_url: Expected URL from ground truth
        
    Returns:
        bool: True if URLs match
    """
    norm_result = normalize_url(result_url)
    norm_expected = normalize_url(expected_url)
    
    if not norm_result or not norm_expected:
        return False
    
    # Exact match
    if norm_result == norm_expected:
        return True
    
    # Check if one is a substring of the other (handles url vs url_preview)
    if norm_result in norm_expected or norm_expected in norm_result:
        return True
    
    return False


def filter_ethz_domains(relevant_docs):
    """
    Filter relevant docs to only include ethz.ch main domain (not subdomains).
    
    Args:
        relevant_docs: List of URLs
        
    Returns:
        List of ethz.ch URLs (excluding subdomains like subdomain.ethz.ch)
    """
    ethz_docs = []
    
    for doc in relevant_docs:
        if not doc or not isinstance(doc, str):
            continue
        
        doc = doc.strip()
        if not doc:
            continue
        
        # Parse URL
        parsed = urlparse(doc)
        netloc = parsed.netloc.lower()
        
        # Check if it's exactly ethz.ch (not subdomain.ethz.ch)
        # Valid: ethz.ch, www.ethz.ch
        # Invalid: staffnet.ethz.ch, subdomain.ethz.ch
        if netloc == 'ethz.ch' or netloc == 'www.ethz.ch':
            ethz_docs.append(doc)
    
    return ethz_docs


# ----------------------------
# EVALUATION
# ----------------------------
def evaluate_question(
    question,
    relevant_docs,
    es_config,
    top_k=100,
    use_query_expansion=False,
    use_reranker=False,
    rerank_query=None,
    rerank_top_k=100
):
    """
    Evaluate a single question.

    Args:
        question: The question to ask
        relevant_docs: List of relevant document URLs
        es_config: Elasticsearch configuration
        top_k: Number of results to retrieve
        use_query_expansion: Whether to use query expansion
        use_reranker: Whether to use reranker
        rerank_query: Query to use for reranking (defaults to original question if None)
        rerank_top_k: Number of results to keep after reranking

    Returns:
        dict: Evaluation results
    """
    result = {
        'question': question,
        'relevant_docs': relevant_docs,
        'retrieved_count': 0,
        'found_docs': [],
        'missing_docs': [],
        'success': False,
        'rank_of_first_match': None,
        'all_ranks': [],
        'search_results': [],
        'query_expansion_used': use_query_expansion,
        'reranker_used': use_reranker
    }

    try:
        # Query the system with retry logic
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                search_results = simple_search(
                    query=question,
                    index_name=es_config['index_name'],
                    es_url=es_config['es_url'],
                    top_k=top_k,
                    es_user=es_config['es_user'],
                    es_password=es_config['es_password'],
                    use_query_expansion=use_query_expansion,
                    query_expansion_verbose=False,
                    use_reranker=use_reranker,
                    rerank_query=rerank_query,
                    rerank_top_k=rerank_top_k
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries - 1} after error: {e}")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    raise  # Last attempt failed, propagate error

        result['retrieved_count'] = len(search_results)
        result['search_results'] = search_results
        
        # Check each relevant doc
        for relevant_url in relevant_docs:
            found = False
            
            # Check against all retrieved results
            for rank, search_result in enumerate(search_results, start=1):
                # Check both 'url' and 'url_preview' fields
                result_url = search_result.get('url')
                result_url_preview = search_result.get('url_preview')
                
                if is_url_match(result_url, relevant_url) or \
                   is_url_match(result_url_preview, relevant_url):
                    found = True
                    result['found_docs'].append(relevant_url)
                    result['all_ranks'].append(rank)
                    
                    if result['rank_of_first_match'] is None:
                        result['rank_of_first_match'] = rank
                    break
            
            if not found:
                result['missing_docs'].append(relevant_url)
        
        # Success if AT LEAST ONE relevant doc is found
        result['success'] = len(result['found_docs']) > 0
        
    except Exception as e:
        result['error'] = str(e)
        print(f"Error evaluating question '{question}': {e}")
    
    return result


def load_questions_from_excel(excel_path):
    """
    Load questions and relevant docs from Excel file.
    
    Expected columns:
    - question: The question text
    - relevant_doc_1: First relevant document URL
    - relevant_doc_2: Second relevant document URL (optional)
    
    Returns:
        List of tuples (question, [relevant_docs])
    """
    # Read Excel
    df = pd.read_excel(excel_path)
    
    # Normalize column names (case insensitive)
    df.columns = df.columns.str.strip().str.lower()
    
    # Check required columns
    if 'question' not in df.columns:
        raise ValueError("Excel must have 'question' column")
    
    # Find relevant doc columns
    relevant_cols = []
    for col in df.columns:
        if 'relevant' in col and 'doc' in col:
            relevant_cols.append(col)
    
    if not relevant_cols:
        raise ValueError("Excel must have at least one 'relevant_doc' column")
    
    print(f"Found columns: {list(df.columns)}")
    print(f"Relevant doc columns: {relevant_cols}")
    
    questions_data = []
    for idx, row in df.iterrows():
        question = row.get('question')
        
        # Skip if question is empty
        if pd.isna(question) or not str(question).strip():
            continue
        
        question = str(question).strip()
        
        # Collect all relevant docs from all columns
        all_relevant_docs = []
        for col in relevant_cols:
            doc = row.get(col)
            if not pd.isna(doc) and str(doc).strip():
                all_relevant_docs.append(str(doc).strip())
        
        # Filter to only ethz.ch main domain
        ethz_docs = filter_ethz_domains(all_relevant_docs)
        
        # Only include questions that have at least one ethz.ch relevant doc
        if ethz_docs:
            questions_data.append((question, ethz_docs))
    
    return questions_data


def compute_accuracy_at_k(results, k_value):
    """
    Compute accuracy when considering only top-k results.
    
    Args:
        results: List of evaluation results with search_results
        k_value: Consider only top k results
        
    Returns:
        float: Accuracy (fraction of questions with at least one relevant doc in top-k)
    """
    successful = 0
    
    for result in results:
        if not result.get('search_results'):
            continue
        
        # Only consider top-k results
        top_k_results = result['search_results'][:k_value]
        relevant_docs = result['relevant_docs']
        
        # Check if any relevant doc is in top-k
        found = False
        for relevant_url in relevant_docs:
            for search_result in top_k_results:
                result_url = search_result.get('url')
                result_url_preview = search_result.get('url_preview')
                
                if is_url_match(result_url, relevant_url) or \
                   is_url_match(result_url_preview, relevant_url):
                    found = True
                    break
            if found:
                break
        
        if found:
            successful += 1
    
    accuracy = successful / len(results) if results else 0
    return accuracy


def generate_latex_table(k_values, accuracies, output_path):
    """
    Generate LaTeX table with accuracy results.
    
    Args:
        k_values: List of k values
        accuracies: Dict mapping row_name -> list of accuracies for each k
        output_path: Path to save LaTeX table
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write table header
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l" + "c" * len(k_values) + "}\n")
        f.write("\\hline\n")
        
        # Column headers
        header = "Method & " + " & ".join([f"k={k}" for k in k_values]) + " \\\\\n"
        f.write(header)
        f.write("\\hline\n")
        
        # Data rows
        for row_name, acc_list in accuracies.items():
            # Format percentages and escape % character for LaTeX
            formatted_accs = [f"{acc:.1%}".replace("%", "\\%") for acc in acc_list]
            row = f"{row_name} & " + " & ".join(formatted_accs) + " \\\\\n"
            f.write(row)
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Retrieval accuracy at different k values}\n")
        f.write("\\label{tab:retrieval_accuracy}\n")
        f.write("\\end{table}\n")


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--excel", required=True, help="Path to Excel file with questions")
    parser.add_argument("--top-k", type=int, default=100, 
                        help="Number of results to retrieve (default: 100)")
    parser.add_argument("--output", default=None, 
                        help="Output JSON file for results (default: auto-generated)")
    parser.add_argument("--latex-output", default="output/evaluation/accuracy_table.tex",
                        help="LaTeX table output file (default: output/evaluation/accuracy_table.tex)")
    parser.add_argument("--use-query-expansion", action="store_true",
                        help="Enable query expansion for improved retrieval")
    parser.add_argument("--compare-query-expansion", action="store_true",
                        help="Compare retrieval with and without query expansion (deprecated: use --all-scenarios)")
    parser.add_argument("--use-reranker", action="store_true",
                        help="Enable cross-encoder reranking")
    parser.add_argument("--rerank-top-k", type=int, default=100,
                        help="Number of results to keep after reranking (default: 100)")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Run all 4 evaluation scenarios (baseline, query exp, reranker, query exp + reranker)")
    
    args = parser.parse_args()
    
    # Validate reranker availability
    if args.use_reranker and not RERANKER_AVAILABLE:
        print("ERROR: sentence-transformers is not installed.")
        print("Install it with: pip install sentence-transformers")
        sys.exit(1)
    
    # Load ES configuration
    es_config = {
        'index_name': os.getenv('INDEX_NAME', 'ethz_webarchive'),
        'es_url': os.getenv('ES_URL', 'https://es.swissai.cscs.ch'),
        'es_user': os.getenv('ELASTIC_USERNAME'),
        'es_password': os.getenv('ELASTIC_PASSWORD')
    }
    
    # Validate config
    if not es_config['es_user'] or not es_config['es_password']:
        raise ValueError("ELASTIC_USERNAME and ELASTIC_PASSWORD must be set in .env")
    
    print("=" * 70)
    print("RAG SYSTEM EVALUATION")
    print("=" * 70)
    print(f"Excel file: {args.excel}")
    print(f"Top-K: {args.top_k}")
    print(f"Index: {es_config['index_name']}")
    print(f"ES URL: {es_config['es_url']}")
    print(f"Query Expansion: {args.use_query_expansion or args.compare_query_expansion}")
    print(f"Reranker: {args.use_reranker}")
    if args.use_reranker:
        print(f"Rerank Top-K: {args.rerank_top_k}")
    print("=" * 70)
    print()
    
    # Load questions
    print("Loading questions from Excel...")
    questions_data = load_questions_from_excel(args.excel)
    print(f"Loaded {len(questions_data)} questions with ethz.ch relevant docs")
    print()
    
    if not questions_data:
        print("No questions found with ethz.ch relevant docs!")
        return
    
    # Define evaluation scenarios
    # Each scenario is: (name, use_query_expansion, use_reranker, rerank_query_override)
    # rerank_query_override: None means use the query as-is, "original" means use original question
    all_4_scenarios = [
        ("Retrieval Only", False, False, None),
        ("Query Exp. + Retrieval", True, False, None),
        ("Retrieval + Reranker", False, True, None),  # Uses original query for reranking
        ("Query Exp. + Retrieval + Reranker", True, True, "original"),  # Rerank with original query!
    ]

    # Filter scenarios based on user flags
    if args.all_scenarios or args.compare_query_expansion:
        # Run all 4 scenarios
        scenarios = all_4_scenarios
    elif args.use_query_expansion and args.use_reranker:
        # Only run query expansion + reranker scenario
        scenarios = [("Query Exp. + Retrieval + Reranker", True, True, "original")]
    elif args.use_query_expansion:
        # Only run query expansion scenario
        scenarios = [("Query Exp. + Retrieval", True, False, None)]
    elif args.use_reranker:
        # Only run reranker scenario
        scenarios = [("Retrieval + Reranker", False, True, None)]
    else:
        # Default: run baseline only
        scenarios = [("Retrieval Only", False, False, None)]

    # Store results for each scenario
    all_results = {}
    all_metrics = {}

    # Run each scenario
    for scenario_idx, (scenario_name, use_qe, use_rr, rerank_override) in enumerate(scenarios):
        print("\n" + "=" * 70)
        print(f"SCENARIO {scenario_idx + 1}/{len(scenarios)}: {scenario_name}")
        print("=" * 70)
        print()

        results = []
        successful = 0
        total_relevant_docs = 0
        total_found_docs = 0

        for i, (question, relevant_docs) in enumerate(questions_data, start=1):
            print(f"[{i}/{len(questions_data)}] {question[:80]}...")

            # Determine rerank_query parameter
            # If rerank_override is "original", pass the original question
            # Otherwise, pass None (will use query as-is in simple_search)
            rerank_query_param = question if rerank_override == "original" else None

            result = evaluate_question(
                question,
                relevant_docs,
                es_config,
                args.top_k,
                use_query_expansion=use_qe,
                use_reranker=use_rr,
                rerank_query=rerank_query_param,
                rerank_top_k=args.rerank_top_k
            )
            results.append(result)

            if result['success']:
                successful += 1
                print(f"  ✓ SUCCESS")
            else:
                print(f"  ✗ FAILURE")

            total_relevant_docs += len(relevant_docs)
            total_found_docs += len(result['found_docs'])
            print()
            sys.stdout.flush()

        # Compute metrics for this scenario
        accuracy = successful / len(questions_data) if questions_data else 0
        recall = total_found_docs / total_relevant_docs if total_relevant_docs > 0 else 0
        first_match_ranks = [r['rank_of_first_match'] for r in results
                            if r['rank_of_first_match'] is not None]
        avg_rank = sum(first_match_ranks) / len(first_match_ranks) if first_match_ranks else None

        # Store results
        all_results[scenario_name] = results
        all_metrics[scenario_name] = {
            'successful': successful,
            'failed': len(questions_data) - successful,
            'accuracy': accuracy,
            'recall': recall,
            'avg_rank_first_match': avg_rank
        }
    
    # Compute accuracy at different k values
    k_values = [1, 3, 5, 10, 25, 50, 100]

    print("\n" + "=" * 70)
    print("ACCURACY AT DIFFERENT K VALUES")
    print("=" * 70)

    accuracies_dict = {}
    for scenario_name, results in all_results.items():
        print(f"\n{scenario_name}:")
        scenario_accuracies = []
        for k in k_values:
            acc_at_k = compute_accuracy_at_k(results, k)
            scenario_accuracies.append(acc_at_k)
            print(f"  Accuracy @ k={k:3d}: {acc_at_k:.2%}")
        accuracies_dict[scenario_name] = scenario_accuracies
        # Store in metrics dict
        all_metrics[scenario_name]['accuracy_at_k'] = {
            f'k_{k}': acc for k, acc in zip(k_values, scenario_accuracies)
        }

    print("=" * 70)
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions: {len(questions_data)}")

    for scenario_name, metrics in all_metrics.items():
        print(f"\n{scenario_name}:")
        print(f"  Successful: {metrics['successful']}")
        print(f"  Failed: {metrics['failed']}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        if metrics['avg_rank_first_match']:
            print(f"  Avg rank of first match: {metrics['avg_rank_first_match']:.1f}")

    print("=" * 70)
    
    # Save detailed results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(scenarios) == 4:
            mode_suffix = "_all_scenarios"
        elif len(scenarios) == 1:
            mode_suffix = f"_{scenarios[0][0].lower().replace(' ', '_').replace('+', '').replace('.', '')}"
        else:
            mode_suffix = "_custom"
        output_file = f"output/rag_evaluation/rag_evaluation{mode_suffix}_{timestamp}.json"

    # Generate LaTeX table
    generate_latex_table(k_values, accuracies_dict, args.latex_output)
    print(f"\nLaTeX table saved to: {args.latex_output}")

    # Prepare output data
    output_data = {
        'config': {
            'excel_file': args.excel,
            'top_k': args.top_k,
            'index_name': es_config['index_name'],
            'use_query_expansion': args.use_query_expansion,
            'compare_query_expansion': args.compare_query_expansion,
            'use_reranker': args.use_reranker,
            'rerank_top_k': args.rerank_top_k,
            'scenarios_run': [name for name, _, _, _ in scenarios],
            'timestamp': datetime.now().isoformat()
        },
        'metrics': {
            'total_questions': len(questions_data),
            'scenarios': all_metrics
        },
        'results': all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()

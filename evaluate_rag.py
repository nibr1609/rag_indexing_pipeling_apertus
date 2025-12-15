#!/usr/bin/env python3
"""
RAG System Evaluation Script

Evaluates the RAG system by:
1. Reading questions and relevant docs from Excel
2. Querying the system for each question
3. Checking if relevant docs are in top-K results
4. Computing accuracy metrics

Usage:
    python evaluate_rag.py --excel path/to/questions.xlsx --top-k 100
"""

import os
import sys
import argparse
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import urlparse, urlunparse
from query_elasticsearch import simple_search
from datetime import datetime
import json

# Load environment
load_dotenv()


def normalize_url(url):
    """
    Normalize URL for robust matching.

    Handles:
    - Removes fragments (#...)
    - Removes trailing slashes
    - Removes index.html
    - Lowercases
    - Removes query parameters (optional)

    Examples:
        https://ethz.ch/staffnet/en/page.html#section -> https://ethz.ch/staffnet/en/page.html
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

    # Remove fragment (#...)
    # Remove query parameters (?..)
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

    # Remove /index.html at the end
    if normalized.endswith('/index.html'):
        normalized = normalized[:-11]
    elif normalized.endswith('/index.htm'):
        normalized = normalized[:-10]

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


def evaluate_question(question, relevant_docs, es_config, top_k=100):
    """
    Evaluate a single question.

    Args:
        question: The question to ask
        relevant_docs: List of relevant document URLs
        es_config: Elasticsearch configuration
        top_k: Number of results to retrieve

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
        'all_ranks': []
    }

    try:
        # Query the system
        search_results = simple_search(
            query=question,
            index_name=es_config['index_name'],
            es_url=es_config['es_url'],
            top_k=top_k,
            es_user=es_config['es_user'],
            es_password=es_config['es_password']
        )

        result['retrieved_count'] = len(search_results)

        # Check each relevant doc
        for relevant_url in relevant_docs:
            found = False

            # Check against all retrieved results
            for rank, search_result in enumerate(search_results, start=1):
                # Check both 'url' and 'url_preview' fields
                result_url = search_result.get('url')
                result_url_preview = search_result.get('url_preview')

                if is_url_match(result_url, relevant_url) or is_url_match(result_url_preview, relevant_url):
                    found = True
                    result['found_docs'].append(relevant_url)
                    result['all_ranks'].append(rank)

                    if result['rank_of_first_match'] is None:
                        result['rank_of_first_match'] = rank

                    break

            if not found:
                result['missing_docs'].append(relevant_url)

        # Success if ALL relevant docs are found
        result['success'] = len(result['missing_docs']) == 0

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--excel", required=True, help="Path to Excel file with questions")
    parser.add_argument("--top-k", type=int, default=100, help="Number of results to retrieve (default: 100)")
    parser.add_argument("--output", default=None, help="Output JSON file for results (default: auto-generated)")

    args = parser.parse_args()

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

    # Evaluate each question
    results = []
    successful = 0
    total_relevant_docs = 0
    total_found_docs = 0

    print("Evaluating questions...")
    print()

    for i, (question, relevant_docs) in enumerate(questions_data, start=1):
        print(f"[{i}/{len(questions_data)}] Evaluating: {question[:80]}...")

        result = evaluate_question(question, relevant_docs, es_config, args.top_k)
        results.append(result)

        if result['success']:
            successful += 1
            print(f"  ✓ SUCCESS - Found all {len(relevant_docs)} relevant docs")
        else:
            print(f"  ✗ FAILURE - Found {len(result['found_docs'])}/{len(relevant_docs)} docs")
            print(f"    Missing: {result['missing_docs']}")

        if result['rank_of_first_match']:
            print(f"  First match at rank: {result['rank_of_first_match']}")

        total_relevant_docs += len(relevant_docs)
        total_found_docs += len(result['found_docs'])

        print()

    # Compute metrics
    accuracy = successful / len(questions_data) if questions_data else 0
    recall = total_found_docs / total_relevant_docs if total_relevant_docs > 0 else 0

    # Compute average rank of first match
    first_match_ranks = [r['rank_of_first_match'] for r in results if r['rank_of_first_match'] is not None]
    avg_rank = sum(first_match_ranks) / len(first_match_ranks) if first_match_ranks else None

    # Summary
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions: {len(questions_data)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(questions_data) - successful}")
    print(f"Accuracy (all docs found): {accuracy:.2%}")
    print(f"Recall (individual docs): {recall:.2%}")
    if avg_rank:
        print(f"Average rank of first match: {avg_rank:.1f}")
    print("=" * 70)

    # Breakdown by success/failure
    print("\nSUCCESSFUL QUESTIONS:")
    for result in results:
        if result['success']:
            print(f"  - {result['question'][:80]}")

    print("\nFAILED QUESTIONS:")
    for result in results:
        if not result['success']:
            print(f"  - {result['question'][:80]}")
            print(f"    Missing: {result['missing_docs']}")

    # Save detailed results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output/rag_evalutation/rag_evaluation_{timestamp}.json"

    output_data = {
        'config': {
            'excel_file': args.excel,
            'top_k': args.top_k,
            'index_name': es_config['index_name'],
            'timestamp': datetime.now().isoformat()
        },
        'metrics': {
            'total_questions': len(questions_data),
            'successful': successful,
            'failed': len(questions_data) - successful,
            'accuracy': accuracy,
            'recall': recall,
            'avg_rank_first_match': avg_rank
        },
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()

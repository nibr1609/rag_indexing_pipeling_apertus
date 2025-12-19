#!/usr/bin/env python3
"""
Simple script to query the ETH Web Archive index.
Reads configuration from .env file.
"""

import os
from dotenv import load_dotenv
from query_elasticsearch import simple_search, print_search_results

# Load environment variables
load_dotenv()

# Read configuration from environment
index_name = os.getenv('INDEX_NAME', 'ethz_webarchive')
es_url = os.getenv('ES_URL', 'https://es.swissai.cscs.ch')
embedding_service_url = os.getenv('EMBEDDING_SERVICE_URL')
es_username = os.getenv('ELASTIC_USERNAME')
es_password = os.getenv('ELASTIC_PASSWORD')

# Validate credentials for remote servers
is_local = "127.0.0.1" in es_url or "localhost" in es_url
if not is_local and (not es_username or not es_password):
    raise ValueError(
        "Elasticsearch credentials are required for remote servers.\n"
        "Please set ELASTIC_USERNAME and ELASTIC_PASSWORD in your .env file."
    )

# Validate embedding service URL
if not embedding_service_url:
    raise ValueError(
        "EMBEDDING_SERVICE_URL is required.\n"
        "Please set EMBEDDING_SERVICE_URL in your .env file."
    )


def main():
    """
    Interactive query interface for the web archive.
    """
    print("=" * 70)
    print("ETH Web Archive Semantic Search")
    print("=" * 70)
    print(f"Index: {index_name}")
    print(f"ES URL: {es_url}")
    print(f"Embedding Service: {embedding_service_url}")
    print("=" * 70)
    print()

    query = input("Enter your query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    print()
    top_k_input = input("Number of results (default: 5): ").strip()
    top_k = int(top_k_input) if top_k_input else 5

    print()
    print("Executing semantic search...")
    print()

    results = simple_search(
        query=query,
        index_name=index_name,
        es_url=es_url,
        top_k=top_k,
        es_user=es_username,
        es_password=es_password
    )

    print_search_results(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nQuery cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the embedding service is accessible")
        print("2. Check your .env file has correct credentials and EMBEDDING_SERVICE_URL")
        print("3. Verify the embedding service endpoint is running")

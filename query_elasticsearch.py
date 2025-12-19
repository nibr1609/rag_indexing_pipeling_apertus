#!/usr/bin/env python3
"""
Query Elasticsearch index with natural language queries.
Returns relevant documents with metadata including URLs and retrieval dates.
"""

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.llms.ollama import Ollama
from remote_embedding import RemoteEmbedding
from query_expansion import expand_query
import os

# TODO

# def query_webarchive(
#     query,
#     index_name="ethz_webarchive",
#     es_url="https://es.swissai.cscs.ch",
#     embedding_model="nomic-embed-text",
#     llm_model="llama3.2",
#     top_k=5,
#     return_metadata=True,
#     verbose=True,
#     es_user=None,
#     es_password=None
# ):
#     """
#     Query the Elasticsearch web archive index.

#     Args:
#         query (str): Natural language query
#         index_name (str): Name of Elasticsearch index to query
#         es_url (str): Elasticsearch URL
#         embedding_model (str): Ollama embedding model name
#         llm_model (str): Ollama LLM model for response generation
#         top_k (int): Number of top results to retrieve
#         return_metadata (bool): Whether to return full metadata with results
#         verbose (bool): Whether to print debug information
#         es_user (str, optional): Elasticsearch username (required for remote servers)
#         es_password (str, optional): Elasticsearch password (required for remote servers)

#     Returns:
#         dict: Dictionary with 'response', 'source_nodes', and 'metadata'
#             - response: Generated answer from LLM
#             - source_nodes: List of retrieved document chunks
#             - metadata: List of metadata dicts with url, retrieval_date, domain, etc.
#     """
#     if verbose:
#         print("=" * 70)
#         print("Querying Web Archive")
#         print("=" * 70)
#         print(f"Query: {query}")
#         print(f"Index: {index_name}")
#         print(f"Top K: {top_k}")
#         print("=" * 70)

#     # Validate credentials for remote servers
#     is_local = "127.0.0.1" in es_url or "localhost" in es_url
#     if not is_local and (not es_user or not es_password):
#         raise ValueError(
#             "Elasticsearch credentials are required for remote servers.\n"
#             "Please provide es_user and es_password parameters, or set\n"
#             "ELASTIC_USERNAME and ELASTIC_PASSWORD in your .env file."
#         )

#     # Connect to Elasticsearch vector store
#     if is_local:
#         es_vector_store = ElasticsearchStore(
#             index_name=index_name,
#             vector_field='doc_vector',
#             text_field='content',
#             es_url=es_url
#         )
#     else:
#         es_vector_store = ElasticsearchStore(
#             index_name=index_name,
#             vector_field='doc_vector',
#             text_field='content',
#             es_url=es_url,
#             es_user=es_user,
#             es_password=es_password
#         )

#     # Create embedding model
#     embed_model = OllamaEmbedding(embedding_model)

#     # Create LLM
#     llm = Ollama(model=llm_model, request_timeout=120.0)

#     # Create index from vector store
#     index = VectorStoreIndex.from_vector_store(
#         vector_store=es_vector_store,
#         embed_model=embed_model
#     )

#     # Create query engine
#     query_engine = index.as_query_engine(
#         llm=llm,
#         similarity_top_k=top_k,
#         response_mode="compact"
#     )

#     # Execute query
#     if verbose:
#         print("\nExecuting query...")

#     response = query_engine.query(query)

#     # Extract metadata from source nodes
#     metadata_list = []
#     if return_metadata and response.source_nodes:
#         for node in response.source_nodes:
#             node_metadata = {
#                 "score": node.score if hasattr(node, 'score') else None,
#                 "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
#                 "url": node.metadata.get('url'),
#                 "url_preview": node.metadata.get('url_preview'),
#                 "retrieval_date": node.metadata.get('retrieval_date'),
#                 "domain": node.metadata.get('domain'),
#                 "title": node.metadata.get('title'),
#                 "file_path": node.metadata.get('file_path'),
#             }
#             metadata_list.append(node_metadata)

#     if verbose:
#         print("\n" + "=" * 70)
#         print("Response:")
#         print("=" * 70)
#         print(str(response))
#         print("\n" + "=" * 70)
#         print(f"Sources: {len(response.source_nodes)} documents")
#         print("=" * 70)

#         if metadata_list:
#             for i, meta in enumerate(metadata_list, 1):
#                 print(f"\n[{i}] {meta.get('title', 'Untitled')}")
#                 print(f"    URL: {meta.get('url', 'N/A')}")
#                 print(f"    URL Preview: {meta.get('url_preview', 'N/A')}")
#                 print(f"    Domain: {meta.get('domain', 'N/A')}")
#                 print(f"    Retrieved: {meta.get('retrieval_date', 'N/A')}")
#                 if meta.get('score'):
#                     print(f"    Score: {meta['score']:.4f}")

#     return {
#         'response': str(response),
#         'source_nodes': response.source_nodes,
#         'metadata': metadata_list
#     }


def simple_search(
    query,
    index_name="ethz_webarchive",
    es_url="https://es.swissai.cscs.ch",
    top_k=5,
    es_user=None,
    es_password=None,
    use_query_expansion=False,
    query_expansion_verbose=False
):
    """
    Simple semantic search without LLM response generation.
    Just returns the most relevant documents.

    Args:
        query (str): Search query
        index_name (str): Name of Elasticsearch index
        es_url (str): Elasticsearch URL
        top_k (int): Number of results to return
        es_user (str, optional): Elasticsearch username (required for remote servers)
        es_password (str, optional): Elasticsearch password (required for remote servers)
        use_query_expansion (bool): Whether to expand query before search (default: False)
        query_expansion_verbose (bool): Whether to print query expansion details (default: False)

    Returns:
        list: List of dicts with document text and metadata
    """
    # Expand query if requested
    original_query = query
    if use_query_expansion:
        try:
            query = expand_query(query, verbose=query_expansion_verbose)
            if query_expansion_verbose:
                print(f"\n[Query Expansion]")
                print(f"  Original: {original_query}")
                print(f"  Expanded: {query}")
        except Exception as e:
            print(f"Warning: Query expansion failed ({e}), using original query")
            query = original_query
    # Validate credentials for remote servers
    is_local = "127.0.0.1" in es_url or "localhost" in es_url
    if not is_local and (not es_user or not es_password):
        raise ValueError(
            "Elasticsearch credentials are required for remote servers.\n"
            "Please provide es_user and es_password parameters, or set\n"
            "ELASTIC_USERNAME and ELASTIC_PASSWORD in your .env file."
        )

    # Connect to Elasticsearch vector store with extended timeout
    # Configure client options with longer timeout for remote connections
    from elasticsearch import AsyncElasticsearch

    # Create ES client with proper timeout configuration
    if is_local:
        es_client = AsyncElasticsearch(
            hosts=[es_url],
            request_timeout=120,  # 120 seconds for request timeout
            max_retries=3,
            retry_on_timeout=True
        )
        es_vector_store = ElasticsearchStore(
            index_name=index_name,
            vector_field='doc_vector',
            text_field='content',
            es_client=es_client
        )
    else:
        es_client = AsyncElasticsearch(
            hosts=[es_url],
            basic_auth=(es_user, es_password),
            request_timeout=120,  # 120 seconds for request timeout
            max_retries=3,
            retry_on_timeout=True
        )
        es_vector_store = ElasticsearchStore(
            index_name=index_name,
            vector_field='doc_vector',
            text_field='content',
            es_client=es_client
        )

    # Create remote embedding model
    embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL")
    if not embedding_service_url:
        raise ValueError("EMBEDDING_SERVICE_URL not set in environment variables")

    embed_model = RemoteEmbedding(
        service_url=embedding_service_url,
        timeout=300.0
    )

    # Create index from vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        embed_model=embed_model
    )

    # Create retriever
    retriever = index.as_retriever(similarity_top_k=top_k)

    # Execute search
    nodes = retriever.retrieve(query)

    # Format results
    results = []
    for i, node in enumerate(nodes, 1):
        result = {
            "rank": i,
            "score": node.score if hasattr(node, 'score') else None,
            "text": node.text,
            "url": node.metadata.get('url'),
            "url_preview": node.metadata.get('url_preview'),
            "retrieval_date": node.metadata.get('retrieval_date'),
            "domain": node.metadata.get('domain'),
            "title": node.metadata.get('title'),
            "file_path": node.metadata.get('file_path'),
        }
        results.append(result)

    # Add metadata about query expansion to first result (if any)
    if results and use_query_expansion:
        results[0]['_query_expansion_used'] = True
        results[0]['_original_query'] = original_query
        results[0]['_expanded_query'] = query

    return results


def print_search_results(results):
    """
    Pretty print search results.

    Args:
        results (list): List of search result dicts
    """
    print("=" * 70)
    print(f"Found {len(results)} results")
    print("=" * 70)

    for result in results:
        print(f"\n[{result['rank']}] {result.get('title', 'Untitled')}")
        print(f"    URL: {result.get('url', 'N/A')}")
        print(f"    URL Preview (browser-friendly): {result.get('url_preview', 'N/A')}")
        print(f"    Domain: {result.get('domain', 'N/A')}")
        print(f"    Retrieved: {result.get('retrieval_date', 'N/A')}")
        if result.get('score'):
            print(f"    Score: {result['score']:.4f}")
        print(f"    Preview: {result['text'][:150]}...")


# def main():
#     """Example usage."""
#     # Example 1: Query with LLM response
#     print("Example 1: Query with LLM-generated response")
#     result = query_webarchive(
#         query="What research projects are happening at ETH Zurich?",
#         index_name="ethz_webarchive",
#         top_k=3
#     )

#     print("\n\n" + "=" * 70)
#     print("Example 2: Simple semantic search")
#     print("=" * 70)

#     # Example 2: Simple search
#     results = simple_search(
#         query="machine learning courses",
#         index_name="ethz_webarchive",
#         top_k=5
#     )

#     print_search_results(results)


# if __name__ == "__main__":
#     main()

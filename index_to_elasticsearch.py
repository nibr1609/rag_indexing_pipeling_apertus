#!/usr/bin/env python3
"""
Index markdown files to Elasticsearch with embeddings.
Includes URL and retrieval date metadata from domain mappings.
"""

import json
import requests
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from remote_embedding import RemoteEmbedding
import os
import warnings


def extract_timestamp_from_path(file_path):
    """
    Extract timestamp from file path if it contains WARC filename pattern.
    Returns datetime object or None.

    Pattern: YYYYMMDDHHMMSS in WARC filename
    Example: ARCHIVEIT-19945-TEST-JOB2538000-0-SEED4432727-20250409125201867-00000-9618ziof.warc.gz

    Note: After combining domains, timestamps are lost. This will return None.
    To preserve timestamps, you need to store them during the combine step.
    """
    # Look for pattern: 14 digits in the path
    match = re.search(r'-(\d{14})\d*-', str(file_path))
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
        except ValueError:
            pass
    return None


def load_domain_mappings(mappings_path):
    """
    Load domain to URL mappings from JSON file.

    Args:
        mappings_path (str): Path to domain mappings JSON file

    Returns:
        dict: Domain to URL mapping
    """
    if not mappings_path or not Path(mappings_path).exists():
        print(f"Warning: Mappings file not found at {mappings_path}")
        return {}

    try:
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        print(f"Loaded {len(mappings)} domain mappings")
        return mappings
    except Exception as e:
        print(f"Error loading mappings: {e}")
        return {}


def load_timestamps(timestamps_path):
    """
    Load file retrieval timestamps from JSON file.

    Args:
        timestamps_path (str): Path to timestamps JSON file

    Returns:
        dict: File path to timestamp mapping
    """
    if not timestamps_path or not Path(timestamps_path).exists():
        print(f"Warning: Timestamps file not found at {timestamps_path}")
        return {}

    try:
        with open(timestamps_path, 'r', encoding='utf-8') as f:
            timestamps = json.load(f)
        print(f"Loaded timestamps for {len(timestamps)} files")
        return timestamps
    except Exception as e:
        print(f"Error loading timestamps: {e}")
        return {}


def get_documents_from_markdown_files(markdown_dir, domain_mappings=None, timestamps=None):
    """
    Reads markdown files and returns list of LlamaIndex Documents with metadata.

    Args:
        markdown_dir (str): Directory containing markdown files organized by domain
        domain_mappings (dict): Optional mapping of domain to original URL
        timestamps (dict): Optional mapping of file paths to retrieval timestamps

    Returns:
        list: List of Document objects with metadata
    """
    documents = []
    markdown_path = Path(markdown_dir)

    if not markdown_path.exists():
        print(f"Error: Directory {markdown_dir} does not exist")
        return documents

    # Find all markdown files recursively
    md_files = list(markdown_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    for md_file in tqdm(md_files, desc="Loading markdown files"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Skip empty files
            if not content.strip():
                continue

            # Get relative path from markdown_dir
            relative_path = md_file.relative_to(markdown_path)

            # Extract domain (first directory in path)
            domain = relative_path.parts[0] if relative_path.parts else "unknown"

            # Get URL from mappings
            url = None
            url_preview = None
            if domain_mappings and domain in domain_mappings:
                base_url = domain_mappings[domain]
                # Construct full URL by appending path
                if len(relative_path.parts) > 1:
                    # Remove domain and .md extension, reconstruct path
                    page_path = '/'.join(relative_path.parts[1:])
                    page_path = page_path.replace('.md', '.html')
                    if not base_url.endswith('/'):
                        base_url += '/'

                    # url: exact file path with index.html (original archive path)
                    url = base_url + page_path

                    # url_preview: clean version for browsers (remove /index.html)
                    if page_path.endswith('/index.html'):
                        clean_path = page_path[:-11]  # Remove '/index.html'
                        url_preview = base_url + clean_path if clean_path else base_url.rstrip('/')
                    elif page_path == 'index.html':
                        url_preview = base_url.rstrip('/')
                    else:
                        url_preview = url
                else:
                    url = base_url
                    url_preview = base_url

            # Extract retrieval date from timestamps JSON if available
            retrieval_date_str = None
            if timestamps:
                # Convert markdown path to HTML path for lookup
                # markdown: domain/path/to/file.md -> HTML: domain/path/to/file.html
                html_relative_path = str(relative_path).replace('.md', '.html')
                retrieval_date_str = timestamps.get(html_relative_path)

            # Fallback: try to extract from file path (won't work after combine_domains)
            if not retrieval_date_str:
                retrieval_date = extract_timestamp_from_path(str(md_file))
                retrieval_date_str = retrieval_date.isoformat() if retrieval_date else None

            # Extract title from first heading if available
            title = md_file.stem
            lines = content.split('\n')
            for line in lines:
                if line.strip().startswith('# '):
                    title = line.strip().replace('# ', '')
                    break

            # Create metadata
            metadata = {
                "file_path": str(relative_path),
                "file_name": md_file.name,
                "domain": domain,
                "title": title,
                "source": "ethz-webarchive"
            }

            # Add URL if available
            if url:
                metadata["url"] = url

            # Add URL preview if available
            if url_preview:
                metadata["url_preview"] = url_preview

            # Add retrieval date if available
            if retrieval_date_str:
                metadata["retrieval_date"] = retrieval_date_str

            # Create Document object
            # Ensure all metadata (including retrieval_date) is included in embeddings
            doc = Document(
                text=content,
                metadata=metadata,
                excluded_llm_metadata_keys=[],  # Include all metadata for LLM
                excluded_embed_metadata_keys=[]  # Include all metadata in embeddings
            )
            documents.append(doc)

        except Exception as e:
            print(f"Error processing {md_file}: {e}")
            continue

    return documents


def save_documents_to_json(documents, output_file="indexed_documents.json"):
    """
    Save documents to JSON file for inspection/backup.

    Args:
        documents (list): List of Document objects
        output_file (str): Path to output JSON file
    """
    json_docs = []

    for idx, doc in enumerate(documents):
        json_docs.append({
            "doc_id": idx,
            "file_path": doc.metadata.get('file_path', ''),
            "file_name": doc.metadata.get('file_name', ''),
            "domain": doc.metadata.get('domain', ''),
            "title": doc.metadata.get('title', ''),
            "url": doc.metadata.get('url', ''),
            "url_preview": doc.metadata.get('url_preview', ''),
            "retrieval_date": doc.metadata.get('retrieval_date', ''),
            "content": doc.text,
            "source": doc.metadata.get('source', 'ethz-webarchive')
        })

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_docs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(json_docs)} documents to {output_file}")


def clean_elasticsearch_index(index_name, es_url="https://es.swissai.cscs.ch", username="lsaie-1", password=None):
    """
    Delete existing Elasticsearch index to ensure clean state.

    Args:
        index_name (str): Name of the index to delete
        es_url (str): Elasticsearch URL
        username (str): Elasticsearch username
        password (str, optional): Elasticsearch password
    """
    # Only require authentication for remote servers (not localhost)
    auth = None
    if "127.0.0.1" not in es_url and "localhost" not in es_url:
        if not password:
            print("Warning: Elasticsearch password not provided, skipping index deletion")
            return
        auth = (username, password)

    try:
        response = requests.delete(f"{es_url}/{index_name}", auth=auth)
        if response.status_code == 200:
            print(f"Deleted existing index: {index_name}")
        elif response.status_code == 404:
            print(f"Index {index_name} does not exist (this is fine for first run)")
        else:
            print(f"Warning: Could not delete index. Status: {response.status_code}")
    except Exception as e:
        print(f"Warning: Error cleaning index: {e}")


def index_markdown_to_elasticsearch(
    markdown_dir,
    index_name="ethz_webarchive",
    es_url="https://es.swissai.cscs.ch",
    mappings_path=None,
    timestamps_path=None,
    embedding_model="nomic-embed-text",
    chunk_size=512,
    chunk_overlap=64,
    clean_index=True,
    save_json=True,
    json_output_path=None,
    es_user="lsaie-1",
    es_password=None
):
    """
    Index markdown files to Elasticsearch with embeddings.

    Args:
        markdown_dir (str): Directory containing markdown files
        index_name (str): Name of Elasticsearch index
        es_url (str): Elasticsearch URL
        mappings_path (str, optional): Path to domain mappings JSON file
        timestamps_path (str, optional): Path to timestamps JSON file from combine_domains
        embedding_model (str): Ollama embedding model name
        chunk_size (int): Size of text chunks for splitting
        chunk_overlap (int): Overlap between chunks
        clean_index (bool): Whether to delete existing index before indexing
        save_json (bool): Whether to save documents to JSON for inspection
        json_output_path (str, optional): Path for JSON output file
        es_user (str): Elasticsearch username
        es_password (str, optional): Elasticsearch password

    Returns:
        dict: Summary with 'documents_loaded', 'documents_indexed', 'index_name'
    """
    print("=" * 70)
    print("Elasticsearch Indexing Pipeline")
    print("=" * 70)
    print(f"Markdown dir: {markdown_dir}")
    print(f"Index name:   {index_name}")
    print(f"ES URL:       {es_url}")
    print(f"Embedding:    {embedding_model}")
    print("=" * 70)

    # Validate credentials for remote servers
    is_local = "127.0.0.1" in es_url or "localhost" in es_url
    if not is_local:
        if not es_user or not es_password:
            raise ValueError(
                "Elasticsearch credentials are required for remote servers.\n"
                "Please provide es_user and es_password parameters, or set\n"
                "ELASTIC_USERNAME and ELASTIC_PASSWORD in your .env file."
            )

    # Clean index if requested
    if clean_index:
        clean_elasticsearch_index(index_name, es_url, es_user, es_password)

    # Load domain mappings if provided
    domain_mappings = None
    if mappings_path:
        domain_mappings = load_domain_mappings(mappings_path)

    # Load timestamps if provided
    timestamps = None
    if timestamps_path:
        timestamps = load_timestamps(timestamps_path)

    # Load documents from markdown files
    print("\nLoading markdown documents...")
    documents = get_documents_from_markdown_files(markdown_dir, domain_mappings, timestamps)

    if not documents:
        print("No documents found!")
        return {
            'documents_loaded': 0,
            'documents_indexed': 0,
            'index_name': index_name
        }

    print(f"Loaded {len(documents)} documents")

    # Save to JSON if requested
    if save_json:
        if json_output_path is None:
            json_output_path = f"output/{index_name}_documents.json"
        save_documents_to_json(documents, json_output_path)

    # Create Elasticsearch vector store
    # Handle authentication for remote servers
    # Use small batch_size to avoid 413 Request Entity Too Large errors
    if "127.0.0.1" in es_url or "localhost" in es_url:
        es_vector_store = ElasticsearchStore(
            index_name=index_name,
            vector_field="doc_vector",
            text_field="content",
            es_url=es_url,
            batch_size=10  # Limit bulk operations to 10 nodes at a time
        )
    else:
        es_vector_store = ElasticsearchStore(
            index_name=index_name,
            vector_field="doc_vector",
            text_field="content",
            es_url=es_url,
            es_user=es_user,
            es_password=es_password,
            batch_size=10  # Limit bulk operations to 10 nodes at a time
        )

    # Create remote embedding model
    print("\nInitializing remote embedding service...")

    # Get embedding service URL from environment
    embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL")
    if not embedding_service_url:
        raise ValueError("EMBEDDING_SERVICE_URL not set in environment variables")

    try:
        remote_embedding = RemoteEmbedding(
            service_url=embedding_service_url,
            timeout=300.0  # 5 minutes timeout
        )
        print(f"✓ Connected to remote embedding service: {embedding_service_url}")
    except Exception as e:
        print(f"✗ Error connecting to remote embedding service: {e}")
        raise

    # Create ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            remote_embedding,
        ],
        vector_store=es_vector_store
    )

    # Run pipeline with batch processing
    print(f"\nProcessing {len(documents)} documents...")
    print("This may take a while depending on document size and embedding model...")

    # Process in smaller batches to avoid connection timeouts and ES request size limits
    batch_size = 3
    total_processed = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(documents) + batch_size - 1) // batch_size

        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        pipeline.run(documents=batch, show_progress=False)
        total_processed += len(batch)
        print(f"✓ Batch {batch_num} completed successfully ({total_processed}/{len(documents)} documents)")

    print("\n" + "=" * 70)
    print(f"✓ Successfully indexed {len(documents)} documents")
    print(f"✓ Index name: {index_name}")
    print(f"✓ Elasticsearch URL: {es_url}")
    print("=" * 70)

    # Close the Elasticsearch client to prevent unclosed session warnings
    try:
        # Try different methods to close the client
        if hasattr(es_vector_store, 'close'):
            es_vector_store.close()

        # Access the underlying elasticsearch client
        if hasattr(es_vector_store, '_client'):
            client = es_vector_store._client
            if hasattr(client, 'close'):
                client.close()
            elif hasattr(client, 'transport') and hasattr(client.transport, 'close'):
                client.transport.close()

        # Suppress ResourceWarning about unclosed sessions if cleanup fails
        warnings.filterwarnings('ignore', category=ResourceWarning, message='unclosed')
    except Exception as e:
        # Suppress the warning if we can't close properly
        warnings.filterwarnings('ignore', category=ResourceWarning, message='unclosed')
        print(f"Note: Applying workaround for ES client cleanup")

    return {
        'documents_loaded': len(documents),
        'documents_indexed': len(documents),
        'index_name': index_name
    }

#!/usr/bin/env python3
"""
Index markdown files to Elasticsearch with embeddings.
Includes URL and retrieval date metadata from domain mappings.

INCREMENTAL INDEXING:
To avoid re-processing already indexed files, use the indexed_files_path parameter.
The tracking file will be created automatically if not provided (default: output/{index_name}_indexed_files.json).

Direct usage:
    index_markdown_to_elasticsearch(
        markdown_dir="output/markdown/19945",
        indexed_files_path="/path/to/indexed_files.json",  # Optional
        ...
    )

Via pipeline script:
    python run_indexing_pipeline.py \\
        --warc-input-dir ./data/warcs \\
        --topics-excel-path ./topics.xlsx \\
        --indexed-files-path /path/to/indexed_files.json  # Optional

Via SLURM:
    sbatch index.sbatch <warc_dir> <excel_path> [indexed_files_path]

The tracking file stores relative paths of successfully indexed markdown files.
On subsequent runs, files in the tracking list will be skipped with an 'already_indexed' counter.
"""

import json
import requests
import re
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline, run_transformations
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


def load_indexed_files(indexed_files_path):
    """
    Load set of already indexed file paths from JSON file.

    Args:
        indexed_files_path (str): Path to indexed files tracking JSON file

    Returns:
        set: Set of file paths that have been successfully indexed
    """
    if not indexed_files_path or not Path(indexed_files_path).exists():
        print(f"No indexed files tracking found at {indexed_files_path} (starting fresh)")
        return set()

    try:
        with open(indexed_files_path, 'r', encoding='utf-8') as f:
            indexed_files = json.load(f)
        indexed_set = set(indexed_files)
        print(f"Loaded {len(indexed_set)} already indexed files")
        return indexed_set
    except Exception as e:
        print(f"Error loading indexed files: {e}")
        return set()


def save_indexed_files(indexed_files_set, indexed_files_path):
    """
    Save set of successfully indexed file paths to JSON file.

    Args:
        indexed_files_set (set): Set of file paths that have been successfully indexed
        indexed_files_path (str): Path to save indexed files tracking JSON file
    """
    try:
        indexed_list = sorted(list(indexed_files_set))
        output_path = Path(indexed_files_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(indexed_files_path, 'w', encoding='utf-8') as f:
            json.dump(indexed_list, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(indexed_list)} indexed files to {indexed_files_path}")
    except Exception as e:
        print(f"Error saving indexed files: {e}")


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


def get_documents_from_markdown_files(markdown_dir, domain_mappings=None, timestamps=None, force_domain=None, base_path=None, indexed_files=None):
    """
    Reads markdown files and returns list of LlamaIndex Documents.
    STRICTLY FILTERS: Any single line > 1000 characters is deleted.

    Args:
        markdown_dir (str): Directory containing markdown files organized by domain
        domain_mappings (dict): Optional mapping of domain to original URL
        timestamps (dict): Optional mapping of file paths to retrieval timestamps
        force_domain (str): Optional domain to use instead of extracting from path (useful for subdirectories)
        base_path (str): Optional base path to prepend to URLs (e.g., "staffnet/de" when indexing a subdirectory)
        indexed_files (set): Optional set of already indexed file paths to skip

    Returns:
        tuple: (List of Document objects with metadata, already_indexed_count)
    """
    documents = []
    markdown_path = Path(markdown_dir)

    if not markdown_path.exists():
        print(f"Error: Directory {markdown_dir} does not exist")
        return documents, 0

    md_files = list(markdown_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    # Counter for deleted lines to track how much "junk" we removed
    deleted_lines_count = 0
    already_indexed_count = 0

    # Debug: Show sample of indexed files if present
    if indexed_files and len(indexed_files) > 0:
        sample_indexed = list(indexed_files)[:3]
        print(f"Sample of indexed files to skip: {sample_indexed}")

    debug_first_files = 0  # Counter for debug output

    for md_file in tqdm(md_files, desc="Loading markdown files"):
        try:
            # Get relative path first for checking against indexed files
            relative_path = md_file.relative_to(markdown_path)
            relative_path_str = str(relative_path)

            # Extract domain (use force_domain if provided, otherwise first directory in path)
            if force_domain:
                domain = force_domain
                # When force_domain is used, create tracking key with domain prefix
                tracking_key = f"{domain}/{relative_path_str}"
            else:
                domain = relative_path.parts[0] if relative_path.parts else "unknown"
                # Normal case: relative path already includes domain
                tracking_key = relative_path_str

            # Debug: Show tracking keys for first 3 files
            if debug_first_files < 3:
                print(f"Debug file {debug_first_files}: tracking_key='{tracking_key}', in_indexed={tracking_key in indexed_files if indexed_files else False}")
                debug_first_files += 1

            # Check if this file was already indexed
            if indexed_files and tracking_key in indexed_files:
                already_indexed_count += 1
                continue

            with open(md_file, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            # ---------------------------------------------------------
            # 🔴 AGGRESSIVE FILTER: Remove lines > 1000 chars
            # ---------------------------------------------------------
            lines = raw_content.split('\n')
            clean_lines = []
            for line in lines:
                if len(line) > 1000:
                    deleted_lines_count += 1
                    continue  # Skip this massive line entirely
                clean_lines.append(line)

            content = '\n'.join(clean_lines)

            # Skip empty files (or files that became empty after filtering)
            if not content.strip():
                continue
            # ---------------------------------------------------------

            # Get URL from mappings
            url = None
            url_preview = None
            if domain_mappings and domain in domain_mappings:
                base_url = domain_mappings[domain]

                # Determine page_path based on whether we're using force_domain
                if force_domain:
                    # When using force_domain, use the entire relative path
                    # because the domain is not in the path structure
                    page_path = str(relative_path).replace('.md', '.html')
                    # If base_path is provided, prepend it
                    if base_path:
                        page_path = base_path.rstrip('/') + '/' + page_path
                elif len(relative_path.parts) > 1:
                    # Normal case: skip the first part (domain) and use the rest
                    page_path = '/'.join(relative_path.parts[1:])
                    page_path = page_path.replace('.md', '.html')
                else:
                    # Just the domain, no subpath
                    url = base_url
                    url_preview = base_url
                    page_path = None

                if page_path:
                    if not base_url.endswith('/'):
                        base_url += '/'
                    url = base_url + page_path
                    if page_path.endswith('/index.html'):
                        clean_path = page_path[:-11]
                        url_preview = base_url + clean_path if clean_path else base_url.rstrip('/')
                    elif page_path == 'index.html':
                        url_preview = base_url.rstrip('/')
                    else:
                        url_preview = url

            retrieval_date_str = None
            if timestamps:
                html_relative_path = str(relative_path).replace('.md', '.html')
                retrieval_date_str = timestamps.get(html_relative_path)

            if not retrieval_date_str:
                retrieval_date = extract_timestamp_from_path(str(md_file))
                retrieval_date_str = retrieval_date.isoformat() if retrieval_date else None

            title = md_file.stem
            for line in content.split('\n')[:5]:  # Only check first 5 lines for title
                if line.strip().startswith('# '):
                    title = line.strip().replace('# ', '')
                    break

            metadata = {
                "file_path": tracking_key,  # Use tracking_key for consistent identification
                "file_name": md_file.name,
                "domain": domain,
                "title": title,
                "source": "ethz-webarchive"
            }
            if url:
                metadata["url"] = url
            if url_preview:
                metadata["url_preview"] = url_preview
            if retrieval_date_str:
                metadata["retrieval_date"] = retrieval_date_str

            doc = Document(
                text=content,
                metadata=metadata,
                excluded_llm_metadata_keys=[],
                excluded_embed_metadata_keys=[]
            )
            documents.append(doc)

        except Exception as e:
            print(f"Error processing {md_file}: {e}")
            continue

    print(f"⚠️ Filtered out {deleted_lines_count} massive lines (>1000 chars) during loading.")
    if already_indexed_count > 0:
        print(f"ℹ️ Skipped {already_indexed_count} already indexed files")
    return documents, already_indexed_count


def index_markdown_to_elasticsearch(
    markdown_dir,
    index_name="ethz_webarchive",
    es_url="https://es.swissai.cscs.ch",
    mappings_path=None,
    timestamps_path=None,
    embedding_model="nomic-embed-text",
    chunk_size=512,
    chunk_overlap=64,
    clean_index=False,
    save_json=False,
    json_output_path=None,
    es_user="lsaie-1",
    es_password=None,
    force_domain=None,
    base_path=None,
    indexed_files_path=None
):
    """
    Index markdown files to Elasticsearch with embeddings.
    Uses a safe, decoupled approach: split → embed → upload in slices.

    Args:
        markdown_dir (str): Directory containing markdown files
        index_name (str): Name of Elasticsearch index
        es_url (str): Elasticsearch URL
        mappings_path (str, optional): Path to domain mappings JSON file
        timestamps_path (str, optional): Path to timestamps JSON file from combine_domains
        embedding_model (str): Embedding model name (for display only, actual model is remote)
        chunk_size (int): Size of text chunks for splitting
        chunk_overlap (int): Overlap between chunks
        clean_index (bool): Whether to delete existing index before indexing
        save_json (bool): Whether to save documents to JSON for inspection
        json_output_path (str, optional): Path for JSON output file
        es_user (str): Elasticsearch username
        es_password (str, optional): Elasticsearch password
        force_domain (str, optional): Force a specific domain instead of extracting from path
        base_path (str, optional): Base path to prepend to URLs (e.g., "staffnet/de")
        indexed_files_path (str, optional): Path to JSON file tracking already indexed files

    Returns:
        dict: Summary with 'documents_loaded', 'documents_indexed', 'already_indexed'
    """
    # --- SETUP & LOGGING ---
    print("=" * 70)
    sys.stdout.flush()
    print("Elasticsearch Indexing Pipeline (Safe Mode: Decoupled + Sliced Uploads)")
    print(f"Index name:   {index_name}")
    print(f"Chunk Size:   {chunk_size}")
    print("=" * 70)
    sys.stdout.flush()

    is_local = "127.0.0.1" in es_url or "localhost" in es_url
    if not is_local and (not es_user or not es_password):
        raise ValueError("Elasticsearch credentials required for remote server.")

    if clean_index:
        clean_elasticsearch_index(index_name, es_url, es_user, es_password)

    # --- LOAD INDEXED FILES TRACKING ---
    if not indexed_files_path:
        # Set default path if not provided
        indexed_files_path = f"output/{index_name}_indexed_files.json"
    indexed_files = load_indexed_files(indexed_files_path)

    # --- LOAD DOCUMENTS ---
    domain_mappings = load_domain_mappings(mappings_path) if mappings_path else None
    timestamps = load_timestamps(timestamps_path) if timestamps_path else None

    print("\nLoading documents...")
    sys.stdout.flush()
    documents, already_indexed = get_documents_from_markdown_files(
        markdown_dir, domain_mappings, timestamps, force_domain, base_path, indexed_files
    )
    print(f"Documents loaded: {len(documents)}")
    sys.stdout.flush()

    if not documents:
        print("No documents found!")
        sys.stdout.flush()
        return {'documents_loaded': 0, 'documents_indexed': 0, 'already_indexed': already_indexed}

    # --- DEBUG JSON SAVE ---
    if save_json:
        if json_output_path is None:
            json_output_path = f"output/{index_name}_documents.json"
        print(f"Saving debug JSON to {json_output_path}...")
        sys.stdout.flush()
        save_documents_to_json(documents, json_output_path)
        print("JSON saved.")
        sys.stdout.flush()

    # --- ES CONNECTION ---
    print("\nConnecting to Elasticsearch...")
    sys.stdout.flush()
    if is_local:
        es_vector_store = ElasticsearchStore(
            index_name=index_name,
            vector_field="doc_vector",
            text_field="content",
            es_url=es_url,
            batch_size=50  # Batch 50 nodes per upload for speed
        )
    else:
        es_vector_store = ElasticsearchStore(
            index_name=index_name,
            vector_field="doc_vector",
            text_field="content",
            es_url=es_url,
            es_user=es_user,
            es_password=es_password,
            batch_size=50  # Batch 50 nodes per upload for speed
        )

    # --- EMBEDDING SERVICE ---
    embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL")
    if not embedding_service_url:
        raise ValueError("EMBEDDING_SERVICE_URL not set")
    remote_embedding = RemoteEmbedding(service_url=embedding_service_url, timeout=300.0)
    print(f"✓ Connected to services")
    sys.stdout.flush()

    # --- PIPELINE SETUP (SPLITTER ONLY) ---
    # We purposefully do NOT include remote_embedding here. We run it manually.
    splitter_pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        ]
    )

    doc_batch_size = 5  # Process 5 files at a time for better speed
    total_processed = 0
    total_skipped = 0

    # Intermediate save configuration
    save_interval = 200  # Save tracking file every 200 documents
    last_save_count = 0

    # Track successfully indexed files
    if indexed_files is None:
        indexed_files = set()
    newly_indexed_files = set()

    print(f"\nStarting processing of {len(documents)} documents...")
    sys.stdout.flush()

    # --- MAIN LOOP WITH TQDM ---
    for i in tqdm(range(0, len(documents), doc_batch_size), desc="Indexing", file=sys.stdout, mininterval=5.0):
        if i % 10 == 0:
            print("At iteration " + str(i) + " skipped files: " + str(total_skipped))
        batch = documents[i:i+doc_batch_size]

        try:
            # -------------------------------------------------
            # STEP 1: SPLIT (Local CPU only)
            # -------------------------------------------------
            nodes = run_transformations(
                nodes=batch,
                transformations=splitter_pipeline.transformations,
                show_progress=False
            )

            if not nodes:
                continue

            valid_nodes = []

            # -------------------------------------------------
            # STEP 2: EMBED (Network Call - Batched)
            # -------------------------------------------------
            # Filter out nodes that are too large
            embeddable_nodes = []
            for node in nodes:
                content_str = node.get_content()
                # Safety Check: Size
                if len(content_str) > 500_000:
                    # 500k chars is ~1MB. Too big for embedding model context usually.
                    continue
                embeddable_nodes.append(node)

            # Batch embed all valid nodes at once
            if embeddable_nodes:
                try:
                    texts = [node.get_content() for node in embeddable_nodes]
                    embeddings = remote_embedding._get_text_embeddings(texts)

                    # Assign embeddings to nodes
                    for node, embedding in zip(embeddable_nodes, embeddings):
                        node.embedding = embedding
                        valid_nodes.append(node)

                except Exception as e_embed:
                    print(f"\nBatch embedding failed for doc batch {i}")
                    print(f"error: {e_embed}")
                    # Skip this entire batch if embedding fails
                    continue

            if not valid_nodes:
                print("not valid nodes")
                total_skipped += len(batch)
                continue

            # -------------------------------------------------
            # STEP 3: UPLOAD (Sliced to avoid 413 Body Too Large)
            # -------------------------------------------------
            # We slice the list of nodes into mini-batches of 20.
            # This ensures the JSON payload sent to ES is never > 1MB.
            upload_slice_size = 20

            for u_idx in range(0, len(valid_nodes), upload_slice_size):
                sub_batch = valid_nodes[u_idx : u_idx + upload_slice_size]
                try:
                    es_vector_store.add(sub_batch)
                except Exception as e_upload:
                    print(f"\n⚠️ Upload failed for slice {u_idx} in batch {i}")
                    print(f"error: {e_upload}")
                    sys.stdout.flush()
                    continue  # Try the next slice

            # Track successfully indexed files from this batch
            for doc in batch:
                file_path = doc.metadata.get('file_path')
                if file_path:
                    newly_indexed_files.add(file_path)

            total_processed += len(batch)

            # Intermediate save: save tracking file every save_interval documents
            if total_processed - last_save_count >= save_interval:
                if indexed_files_path and newly_indexed_files:
                    all_indexed_files = indexed_files.union(newly_indexed_files)
                    save_indexed_files(all_indexed_files, indexed_files_path)
                    last_save_count = total_processed
                    print(f"\n💾 Intermediate save: {len(all_indexed_files)} files tracked")
                    sys.stdout.flush()

        except Exception as e:
            print(f"\n❌ CRASH in file index {i}")
            print(f"Error: {e}")
            sys.stdout.flush()
            total_skipped += len(batch)
            continue

    print("\n" + "=" * 70)
    sys.stdout.flush()
    print(f"✓ FINISHED.")
    print(f"  Indexed: {total_processed}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Already indexed: {already_indexed}")
    print("=" * 70)
    sys.stdout.flush()

    # Save the updated indexed files list
    if indexed_files_path and newly_indexed_files:
        all_indexed_files = indexed_files.union(newly_indexed_files)
        save_indexed_files(all_indexed_files, indexed_files_path)

    try:
        if hasattr(es_vector_store, 'close'):
            es_vector_store.close()
    except:
        pass

    return {
        'documents_loaded': len(documents),
        'documents_indexed': total_processed,
        'already_indexed': already_indexed
    }

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
"""

import json
import re
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- LlamaIndex Imports ---
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

# --- Local Import (Your Backend Wrapper) ---
try:
    from remote_embedding import RemoteEmbedding
except ImportError:
    print("❌ Critical Error: Could not import 'RemoteEmbedding'.")
    print("Ensure remote_embedding.py is in the same directory.")
    sys.exit(1)


# --- CONFIGURATION ---
CHUNK_SIZE = 512
OVERLAP = 64
MAX_CHAR_LIMIT = 5000  # <--- FIXED: Was 64 (too small), set to 5000 for safety.

# --- HELPER FUNCTIONS (Must be global for multiprocessing) ---

def get_slurm_cores():
    """Robustly detect available cores in a SLURM allocation."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 32

def clean_garbage_text(text):
    """Aggressive cleaning for PDF artifacts."""
    # Remove long strings of non-whitespace characters (base64 garbage)
    text = re.sub(r'\S{100,}', '', text)
    # Remove excessive repeated characters (e.g. "__________")
    text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)
    return text

def adaptive_get_embeddings(remote_embedding_client, texts):
    """
    Robust embedding fetcher. 
    Recursively splits batches if the backend returns a 400/Context Error.
    """
    if not texts:
        return []

    try:
        # Optimistic: Try the whole batch
        return remote_embedding_client._get_text_embeddings(texts)

    except Exception as e:
        error_msg = str(e).lower()
        
        # Check for context length error (400) or similar
        if "400" in error_msg or "length" in error_msg or "large" in error_msg:
            
            # BASE CASE: A single text is failing
            if len(texts) == 1:
                text = texts[0]
                if len(text) > MAX_CHAR_LIMIT:
                    # Truncate and retry
                    truncated = text[:MAX_CHAR_LIMIT]
                    return adaptive_get_embeddings(remote_embedding_client, [truncated])
                else:
                    # Text is small but still failing? Skip it.
                    raise ValueError(f"Chunk failed permanently: {error_msg}")

            # RECURSIVE CASE: Split batch in half
            mid = len(texts) // 2
            left_texts = texts[:mid]
            right_texts = texts[mid:]
            
            return (adaptive_get_embeddings(remote_embedding_client, left_texts) + 
                    adaptive_get_embeddings(remote_embedding_client, right_texts))
        
        raise e

def worker_process_batch(task_payload):
    """
    The Worker Function (runs in separate process).
    Handles: Load -> Clean -> Metadata/URL Gen -> Split -> Embed.
    Returns: (processed_nodes, skipped_count)
    """
    (file_paths, base_dir, domain_mappings, timestamps, 
     force_domain, base_path) = task_payload

    embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL")
    if not embedding_service_url:
        print(f"❌ [Worker Error] EMBEDDING_SERVICE_URL not set")
        sys.stdout.flush()
        return [], len(file_paths)
    
    # Each process creates its OWN connection pool
    remote_embed = RemoteEmbedding(service_url=embedding_service_url, timeout=300.0)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    
    processed_nodes = [] 
    skipped_count = 0

    for f_path_str in file_paths:
        try:
            f_path = Path(f_path_str)
            base_path_obj = Path(base_dir)
            
            # --- Load & Clean ---
            try:
                relative_path = f_path.relative_to(base_path_obj)
            except ValueError:
                print(f"⚠️ [Skip] Path issue for {f_path.name}")
                sys.stdout.flush()
                skipped_count += 1
                continue 

            relative_path_str = str(relative_path)

            # Domain & Tracking Key
            if force_domain:
                domain = force_domain
                tracking_key = f"{domain}/{relative_path_str}"
            else:
                domain = relative_path.parts[0] if relative_path.parts else "unknown"
                tracking_key = relative_path_str

            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    raw = f.read()
            except Exception as e:
                print(f"❌ [Read Error] {f_path.name}: {e}")
                sys.stdout.flush()
                skipped_count += 1
                continue

            # Fast line filtering
            lines = [L for L in raw.split('\n') if len(L) <= 1000]
            content = '\n'.join(lines)
            content = clean_garbage_text(content)
            
            if not content.strip():
                # Not an error, just empty
                skipped_count += 1
                continue

            # --- Metadata & URL Logic (CRITICAL RESTORATION) ---
            url = None
            url_preview = None
            
            if domain_mappings and domain in domain_mappings:
                base_url = domain_mappings[domain]
                
                # Determine page path
                if force_domain:
                    page_path = str(relative_path).replace('.md', '.html')
                    if base_path:
                        page_path = base_path.rstrip('/') + '/' + page_path
                elif len(relative_path.parts) > 1:
                    page_path = '/'.join(relative_path.parts[1:])
                    page_path = page_path.replace('.md', '.html')
                else:
                    url = base_url
                    url_preview = base_url
                    page_path = None

                # Construct full URL
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

            # Timestamp Logic
            retrieval_date = None
            if timestamps:
                html_key = str(relative_path).replace('.md', '.html')
                retrieval_date = timestamps.get(html_key)

            # Title extraction
            title = f_path.stem
            
            # Metadata Dict
            metadata = {
                "file_path": tracking_key,
                "file_name": f_path.name,
                "domain": domain,
                "title": title,
                "source": "ethz-webarchive"
            }
            if url: metadata["url"] = url
            if url_preview: metadata["url_preview"] = url_preview
            if retrieval_date: metadata["retrieval_date"] = retrieval_date

            doc = Document(text=content, metadata=metadata)

            # --- Split ---
            nodes = splitter.get_nodes_from_documents([doc])
            if not nodes:
                skipped_count += 1
                continue

            # --- Embed (Network IO) ---
            node_texts = [n.get_content() for n in nodes]
            
            try:
                embeddings = adaptive_get_embeddings(remote_embed, node_texts)
                
                if len(embeddings) != len(nodes):
                    print(f"❌ [Embed Error] {f_path.name}: Mismatch (nodes={len(nodes)}, embs={len(embeddings)})")
                    sys.stdout.flush()
                    skipped_count += 1
                    continue 

                valid_nodes = []
                for node, emb in zip(nodes, embeddings):
                    node.embedding = emb
                    valid_nodes.append(node)
                
                processed_nodes.extend(valid_nodes)

            except Exception as e:
                print(f"❌ [Embed Fail] {f_path.name}: {e}")
                sys.stdout.flush()
                skipped_count += 1
                continue

        except Exception as e:
            print(f"❌ [Unknown Error] {f_path_str}: {e}")
            sys.stdout.flush()
            skipped_count += 1
            continue

    return processed_nodes, skipped_count


def extract_timestamp_from_path(file_path):
    """
    Extract timestamp from file path if it contains WARC filename pattern.
    Returns datetime object or None.
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
    """Load domain to URL mappings from JSON file."""
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
    """Load file retrieval timestamps from JSON file."""
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
    """Load set of already indexed file paths from JSON file."""
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
    """Save set of successfully indexed file paths to JSON file."""
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
    """Save documents to JSON file for inspection/backup."""
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
    """Delete existing Elasticsearch index to ensure clean state."""
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
    Includes filtering for massive lines and garbage text.
    """
    documents = []
    markdown_path = Path(markdown_dir)

    if not markdown_path.exists():
        print(f"Error: Directory {markdown_dir} does not exist")
        return documents, 0

    md_files = list(markdown_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    deleted_lines_count = 0
    already_indexed_count = 0

    for md_file in tqdm(md_files, desc="Loading markdown files"):
        try:
            # Get relative path first for checking against indexed files
            relative_path = md_file.relative_to(markdown_path)
            relative_path_str = str(relative_path)

            if force_domain:
                domain = force_domain
                tracking_key = f"{domain}/{relative_path_str}"
            else:
                domain = relative_path.parts[0] if relative_path.parts else "unknown"
                tracking_key = relative_path_str

            if indexed_files and tracking_key in indexed_files:
                already_indexed_count += 1
                continue

            with open(md_file, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            # ---------------------------------------------------------
            # 1. Filter: Remove lines > 1000 chars
            # ---------------------------------------------------------
            lines = raw_content.split('\n')
            clean_lines = []
            for line in lines:
                if len(line) > 1000:
                    deleted_lines_count += 1
                    continue
                clean_lines.append(line)

            content = '\n'.join(clean_lines)
            
            # ---------------------------------------------------------
            # 2. Filter: Clean garbage text (base64, etc)
            # ---------------------------------------------------------
            content = clean_garbage_text(content)

            if not content.strip():
                continue

            # Get URL from mappings
            url = None
            url_preview = None
            if domain_mappings and domain in domain_mappings:
                base_url = domain_mappings[domain]

                if force_domain:
                    page_path = str(relative_path).replace('.md', '.html')
                    if base_path:
                        page_path = base_path.rstrip('/') + '/' + page_path
                elif len(relative_path.parts) > 1:
                    page_path = '/'.join(relative_path.parts[1:])
                    page_path = page_path.replace('.md', '.html')
                else:
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
            for line in content.split('\n')[:5]:
                if line.strip().startswith('# '):
                    title = line.strip().replace('# ', '')
                    break

            metadata = {
                "file_path": tracking_key,
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

    print(f"⚠️ Filtered out {deleted_lines_count} massive lines (>1000 chars).")
    if already_indexed_count > 0:
        print(f"ℹ️ Skipped {already_indexed_count} already indexed files")
    return documents, already_indexed_count


def index_markdown_to_elasticsearch(
    markdown_dir,
    index_name="ethz_webarchive",
    es_url="https://es.swissai.cscs.ch",
    mappings_path=None,
    timestamps_path=None,
    es_user="lsaie-1",
    es_password=None,
    force_domain=None,
    base_path=None,
    indexed_files_path=None,
    **kwargs
):
    # Detect Resources
    num_workers = get_slurm_cores()
    
    print("=" * 70)
    print("🚀 SUPERCOMPUTER PARALLEL INDEXER")
    print(f"   Detected Cores: {num_workers}")
    print("=" * 70)

    # 1. Load Tracking Data
    if not indexed_files_path: 
        indexed_files_path = f"output/{index_name}_indexed_files.json"
    
    indexed_files = set()
    if Path(indexed_files_path).exists():
        with open(indexed_files_path, 'r') as f:
            indexed_files = set(json.load(f))
    print(f"Already indexed: {len(indexed_files)} files")

    domain_mappings = {}
    if mappings_path:
        with open(mappings_path, 'r') as f: domain_mappings = json.load(f)
        
    timestamps = {}
    if timestamps_path:
        with open(timestamps_path, 'r') as f: timestamps = json.load(f)

    # 2. Gather Files
    print("Scanning files...")
    all_md_files = sorted(list(Path(markdown_dir).rglob("*.md")))
    
    files_to_process = []
    base_dir_path = Path(markdown_dir)
    
    # Pre-filter
    for f in all_md_files:
        try:
            rel_p = f.relative_to(base_dir_path)
            key = f"{force_domain}/{rel_p}" if force_domain else str(rel_p)
            if key not in indexed_files:
                files_to_process.append(str(f))
        except:
            pass

    print(f"Files to process: {len(files_to_process)}")
    if not files_to_process: 
        return

    # 3. Initialize ES Connection (Main Process Only)
    es_store = ElasticsearchStore(
        index_name=index_name,
        es_url=es_url,
        es_user=es_user,
        es_password=es_password,
        batch_size=50
    )

    # 4. Create Tasks
    FILES_PER_TASK = 10
    tasks = []
    
    for i in range(0, len(files_to_process), FILES_PER_TASK):
        batch = files_to_process[i:i+FILES_PER_TASK]
        payload = (batch, str(markdown_dir), domain_mappings, timestamps, 
                   force_domain, base_path)
        tasks.append(payload)

    # 5. Execute Parallel Pipeline
    total_indexed_nodes = 0
    total_skipped_files = 0
    newly_indexed_files = set()
    save_counter = 0

    print(f"Dispatching {len(tasks)} tasks to {num_workers} workers...")
    sys.stdout.flush()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all
        futures = {executor.submit(worker_process_batch, t): t for t in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Indexing"):
            try:
                # Unpack result
                nodes, skipped_count = future.result()
                total_skipped_files += skipped_count
                
                if nodes:
                    # Upload to ES (Sliced)
                    SLICE = 20
                    for k in range(0, len(nodes), SLICE):
                        es_store.add(nodes[k:k+SLICE])
                    
                    total_indexed_nodes += len(nodes)

                    # Track success
                    for node in nodes:
                        fp = node.metadata.get('file_path')
                        if fp: newly_indexed_files.add(fp)
                
                # Incremental Save
                save_counter += 1
                if save_counter % 50 == 0:
                    combined = indexed_files.union(newly_indexed_files)
                    with open(indexed_files_path, 'w') as f:
                        json.dump(sorted(list(combined)), f, indent=2)
                    print(f"\n💾 Checkpoint saved. Indexed: {len(combined)}, Skipped so far: {total_skipped_files}")
                    sys.stdout.flush()

            except Exception as e:
                print(f"\n❌ [Batch Crash] Worker failed: {e}")
                sys.stdout.flush()
                # We assume all files in this task failed
                total_skipped_files += FILES_PER_TASK

    # Final Save
    combined = indexed_files.union(newly_indexed_files)
    with open(indexed_files_path, 'w') as f:
        json.dump(sorted(list(combined)), f, indent=2)

    print(f"\n" + "="*70)
    print(f"🎉 DONE.")
    print(f"  Indexed Nodes:  {total_indexed_nodes}")
    print(f"  Files Indexed:  {len(newly_indexed_files)} (Session)")
    print(f"  Files Skipped:  {total_skipped_files}")
    print(f"  Total Tracked:  {len(combined)}")
    print("="*70)
    
    try:
        if hasattr(es_store, 'client'):
            es_store.client.close()
    except:
        pass
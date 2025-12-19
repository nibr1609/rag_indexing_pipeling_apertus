#!/usr/bin/env python3
"""
Convert HTML files from html_combined directory to markdown files.
Filters domains based on Excel sheet and maintains folder structure.
Creates one markdown file per HTML page.
Supports exclusion of specific domains and file keywords.
"""

import os
import json
import gzip
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from html_to_markdown import convert_to_markdown
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import gc


# ==========================================
# Helper Functions
# ==========================================

def get_base_site_from_url(url_in):
    """
    Extracts the base site from the given URL.
    Example: "http://ethz.ch/about/test.png" returns "ethz.ch"
    """
    if "//" not in url_in:
        base_site = url_in
    else:
        url_in_old = url_in
        base_site = url_in.split("//")[1]
        if base_site == "http:":
            print(f"This url is oddly formed: {url_in_old}")
            base_site = url_in_old.split("//")[2]

    # various artefacts found in the warc files
    base_site = base_site.replace("dns:", "")
    base_site = base_site.replace("mailto:", "")
    base_site = base_site.replace("www.", "")
    base_site = base_site.replace("www0.", "")
    base_site = base_site.replace("www1.", "")
    base_site = base_site.replace("www2.", "")
    base_site = base_site.replace("www3.", "")
    base_site = base_site.split(":")[0]
    base_site = base_site.split("/")[0]

    if base_site[-1] == ".":
        base_site = base_site[:-1]

    return base_site


def get_base_url_from_url(url_in):
    """
    Extracts the base URL (protocol + domain) from a full URL.
    Example: "https://ethz.ch/de/about/test.html" returns "https://ethz.ch"
    """
    if "//" not in url_in:
        return url_in

    # Split into protocol and rest
    parts = url_in.split("//")
    protocol = parts[0] + "//"

    # Get domain from the rest (everything before first /)
    rest = parts[1]
    domain = rest.split("/")[0]

    return protocol + domain


def load_allowed_domains(excel_path):
    """
    Load allowed domains from Excel file.
    """
    df = pd.read_excel(excel_path)
    df = df.fillna("")
    urls = list(df["URL"])

    allowed_domains = set()
    for url in urls:
        if url != "":
            base_site = get_base_site_from_url(url)
            allowed_domains.add(base_site)

    print(f"Loaded {len(allowed_domains)} allowed domains from Excel")
    return allowed_domains


def convert_html_to_markdown(html_path, output_path, create_dirs=True):
    """
    Convert a single HTML file to markdown.
    Returns: 'converted', 'skipped', or 'failed'
    """
    try:
        # Read HTML file (handle both regular and gzipped)
        if str(html_path).endswith('.gz'):
            with gzip.open(html_path, 'rb') as f:
                html_content = f.read()
        else:
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except UnicodeDecodeError:
                with open(html_path, 'r', encoding='unicode_escape') as f:
                    html_content = f.read()

        if not html_content:
            return 'skipped'

        # 1. AGGRESSIVE SANITIZATION
        if isinstance(html_content, bytes):
            html_content = html_content.decode('utf-8', errors='ignore')
        else:
            html_content = html_content.encode('utf-8', errors='ignore').decode('utf-8')

        # 2. ROBUST CONVERSION
        try:
            markdown_text = convert_to_markdown(html_content)
        except BaseException:
            # Catches pyo3_runtime.PanicException from Rust
            return 'failed'

        # Skip empty or redirect-only files
        if markdown_text in ("", "Redirecting"):
            return 'skipped'

        # Create output directory if needed
        if create_dirs:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write markdown file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)

        return 'converted'

    except Exception:
        return 'failed'


# ==========================================
# Core Processing Functions
# ==========================================

def process_single_html_file(args):
    """
    Process a single HTML file (for parallel execution).
    Args: (html_file, domain_folder, output_path, exclude_files)
    """
    html_file, domain_folder, output_path, exclude_files = args

    # Check if filename should be skipped based on keywords
    # This filters specific files like 'impressum.html'
    if exclude_files:
        filename_lower = html_file.name.lower()
        for keyword in exclude_files:
            if keyword in filename_lower:
                return ('skipped', 1)

    # Calculate relative path from domain folder
    rel_path = html_file.relative_to(domain_folder)

    # Create output path (change extension to .md)
    if str(html_file).endswith('.html.gz'):
        md_filename = html_file.name.replace('.html.gz', '.md')
    else:
        md_filename = html_file.stem + '.md'

    output_file_path = output_path / domain_folder.name / rel_path.parent / md_filename

    # Convert HTML to markdown
    status = convert_html_to_markdown(html_file, output_file_path, create_dirs=True)
    return (status, 1)


def process_domain_parallel(domain_folder, input_path, output_path, allowed_domains, exclude_files, max_workers=None):
    """
    Process all HTML files in a domain folder in parallel.
    """
    domain = domain_folder.name

    # Check if domain is in allowed list (if filtering is enabled)
    if allowed_domains is not None and domain not in allowed_domains:
        return {'domain': domain, 'converted': 0, 'skipped': 0}

    # Collect all HTML files in this domain folder
    html_files = []
    for html_file in domain_folder.rglob('*'):
        if not html_file.is_file():
            continue
        if html_file.suffix in ['.html', '.htm'] or str(html_file).endswith('.html.gz'):
            html_files.append(html_file)

    if not html_files:
        return {'domain': domain, 'converted': 0, 'skipped': 0, 'failed': 0}

    # Prepare arguments for parallel processing
    # Pass exclude_files down to the worker
    file_args = [(f, domain_folder, output_path, exclude_files) for f in html_files]

    converted = 0
    skipped = 0
    failed = 0

    file_workers = max_workers if max_workers is not None else min(2, mp.cpu_count())

    try:
        with ThreadPoolExecutor(max_workers=file_workers) as executor:
            for status, count in executor.map(process_single_html_file, file_args):
                if status == 'converted':
                    converted += count
                elif status == 'skipped':
                    skipped += count
                else:  # 'failed'
                    failed += count
    except Exception as e:
        print(f"Error in thread pool for domain {domain}: {e}")
        failed += len(file_args) - (converted + skipped + failed)
    finally:
        del html_files
        del file_args
        gc.collect()

    return {'domain': domain, 'converted': converted, 'skipped': skipped, 'failed': failed}


def convert_html_combined_to_markdown(
    input_dir,
    output_dir,
    excel_path=None,
    mappings_path=None,
    exclude_domains=None,     # NEW: List of exact domain names to skip
    exclude_files=None,       # NEW: List of keywords to skip files
    max_file_workers=None
):
    """
    Convert HTML files from html_combined directory to markdown files.
    """
    print("=" * 70)
    print("HTML to Markdown Converter (Parallelized)")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    
    # Initialize exclusions if None
    if exclude_domains is None:
        exclude_domains = []
    if exclude_files is None:
        exclude_files = []

    # Load allowed domains from Excel if provided
    allowed_domains = None
    if excel_path:
        allowed_domains = load_allowed_domains(excel_path)

    # Load full URLs for mapping if needed
    domain_mappings = {}
    if mappings_path and excel_path:
        df = pd.read_excel(excel_path)
        df = df.fillna("")
        urls = list(df["URL"])

        for url in urls:
            if url != "":
                base_site = get_base_site_from_url(url)
                if base_site not in domain_mappings:
                    domain_mappings[base_site] = get_base_url_from_url(url)

        mappings_path_obj = Path(mappings_path)
        mappings_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(domain_mappings, f, indent=2)
        print(f"Saved domain mappings to {mappings_path}")
        
        del df
        del urls
        gc.collect()

    # Process files
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Get all potential folders
    all_domain_folders = [d for d in input_path.iterdir() if d.is_dir()]
    
    # Filter domains based on exclusion list and allowed list
    domain_folders = []
    skipped_count = 0
    
    for d in all_domain_folders:
        # 1. Check Explicit Exclusion (The crashing domains)
        if d.name in exclude_domains:
            print(f"⚠ Skipping excluded domain: {d.name}")
            skipped_count += 1
            continue
            
        # 2. Check Allowed List (Excel filtering)
        if allowed_domains is not None and d.name not in allowed_domains:
            continue
            
        domain_folders.append(d)

    print(f"\nProcessing {len(domain_folders)} domain folders (Skipped {skipped_count} explicitly)...")
    print("=" * 70)

    # Pre-create output directories
    print("Pre-creating output directories...")
    for domain_folder in domain_folders:
        domain_output_dir = output_path / domain_folder.name
        domain_output_dir.mkdir(parents=True, exist_ok=True)

    if max_file_workers is None:
        max_file_workers = min(16, mp.cpu_count())

    print(f"Processing domains sequentially with {max_file_workers} file workers per domain")

    # Clear memory
    del allowed_domains
    if 'domain_mappings' in locals():
        del domain_mappings
    gc.collect()

    results = []
    failed_domains = []

    for domain_folder in tqdm(domain_folders, desc="Processing domains", unit="domain"):
        domain_name = domain_folder.name
        try:
            result = process_domain_parallel(
                domain_folder=domain_folder,
                input_path=input_path,
                output_path=output_path,
                allowed_domains=None, # Already filtered above
                exclude_files=exclude_files, # Pass the file exclusion list
                max_workers=max_file_workers
            )
            results.append(result)
        except Exception as e:
            print(f"\nError processing domain {domain_name}: {e}")
            failed_domains.append(domain_name)
            results.append({'domain': domain_name, 'converted': 0, 'skipped': 0, 'failed': 0})

    # Aggregate results
    domains_processed = {r['domain'] for r in results if r['converted'] > 0 or r['skipped'] > 0 or r['failed'] > 0}
    files_converted = sum(r['converted'] for r in results)
    files_skipped = sum(r['skipped'] for r in results)
    files_failed = sum(r['failed'] for r in results)

    print("\n" + "=" * 70)
    print(f"✓ Processed {len(domains_processed)} domains")
    print(f"✓ Converted {files_converted} files")
    print(f"✓ Skipped {files_skipped} files")
    print(f"⚠ Failed {files_failed} files")
    print("=" * 70)

    return {
        'domains_processed': len(domains_processed),
        'files_converted': files_converted,
        'files_skipped': files_skipped,
        'files_failed': files_failed
    }


# ==========================================
# Main Execution
# ==========================================

def process_html_pipeline():
    """
    Main entry point for the HTML pipeline.
    Configures paths and exclusion lists.
    """
    
    # 1. Configuration Paths
    html_raw_dir = "output/html_raw"        # Adjust if needed
    html_combined_dir = "output/html_combined"
    markdown_output_dir = "output/markdown"
    topics_excel_path = "data/2025-11-20_19945_topics.xlsx"
    html_mappings_path = "output/domain_mappings.json"
    timestamps_json_path = "output/timestamps.json"

    # 2. Define Exclusions
    # Files to skip (substring match in filename)
    files_to_exclude = [
        "impressum", "datenschutz", "kontakt", "robots",
        "imprint", "data-protection", "contact", "copyright",
        "login", "search", "suche"
    ]

    # Domains to skip (exact match of folder name)
    domains_to_exclude = [
        "polyphys.mat.ethz.ch", # Known crasher
        "broken-site.ethz.ch"
    ]

    # 3. Pipeline Steps
    
    # Step A: Combine Domains (Commented out as requested)
    # combine_domains_by_timestamp(
    #     input_dir=html_raw_dir,
    #     output_dir=html_combined_dir,
    #     timestamps_json_path=timestamps_json_path,
    #     excel_path=topics_excel_path
    # )

    # Step B: Convert to Markdown
    print("Starting HTML conversion pipeline...")
    convert_html_combined_to_markdown(
        input_dir=html_combined_dir,
        output_dir=markdown_output_dir,
        excel_path=topics_excel_path,
        mappings_path=html_mappings_path,
        exclude_domains=domains_to_exclude,
        exclude_files=files_to_exclude
    )

if __name__ == "__main__":
    print("Doing html pipeline")
    process_html_pipeline()
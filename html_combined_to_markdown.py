#!/usr/bin/env python3
"""
Convert HTML files from html_combined directory to markdown files.
Filters domains based on Excel sheet and maintains folder structure.
Creates one markdown file per HTML page.
"""

import os
import json
import gzip
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from html_to_markdown import convert_to_markdown
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import gc


def get_base_site_from_url(url_in):
    """
    Extracts the base site from the given URL.
    Example: "http://ethz.ch/about/test.png" returns "ethz.ch"

    Args:
        url_in (str): The url to find the base site for.

    Returns:
        str: Base Url
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

    Args:
        url_in (str): The full URL

    Returns:
        str: Base URL with protocol and domain only
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

    Args:
        excel_path (str): Path to Excel file with URL column

    Returns:
        set: Set of allowed base domains
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

    Args:
        html_path (Path): Path to HTML file
        output_path (Path): Path where markdown should be saved
        create_dirs (bool): Whether to create directories (set False if pre-created)

    Returns:
        bool: True if successful, False otherwise
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

        if not html_content or html_content == "":
            return False

        # Convert to markdown
        markdown_text = convert_to_markdown(str(html_content))

        # Skip empty or redirect-only files
        if markdown_text in ("", "Redirecting"):
            return False

        # Create output directory if needed
        if create_dirs:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write markdown file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)

        return True

    except Exception as e:
        print(f"Error processing {html_path}: {e}")
        return False


def process_single_html_file(args):
    """
    Process a single HTML file (for parallel execution).

    Args:
        args: Tuple of (html_file, domain_folder, output_path, filenames_to_remove)

    Returns:
        tuple: (status, count) where status is 'converted' or 'skipped'
    """
    html_file, domain_folder, output_path, filenames_to_remove = args

    # Check if filename should be skipped
    for keyword in filenames_to_remove:
        if keyword in html_file.name.lower():
            return ('skipped', 1)

    # Calculate relative path from domain folder
    rel_path = html_file.relative_to(domain_folder)

    # Create output path (change extension to .md)
    if str(html_file).endswith('.html.gz'):
        md_filename = html_file.name.replace('.html.gz', '.md')
    else:
        md_filename = html_file.stem + '.md'

    output_file_path = output_path / domain_folder.name / rel_path.parent / md_filename

    # Convert HTML to markdown (create subdirs on-demand to save memory)
    if convert_html_to_markdown(html_file, output_file_path, create_dirs=True):
        return ('converted', 1)
    else:
        return ('skipped', 1)


def process_domain_parallel(domain_folder, input_path, output_path, allowed_domains, filenames_to_remove, max_workers=None):
    """
    Process all HTML files in a domain folder in parallel.

    Args:
        domain_folder (Path): Domain folder to process
        input_path (Path): Input base path
        output_path (Path): Output base path
        allowed_domains (set): Set of allowed domains (or None)
        filenames_to_remove (list): Keywords to skip
        max_workers (int, optional): Number of thread workers for file I/O. If None, uses CPU count.

    Returns:
        dict: Results with 'domain', 'converted', 'skipped' counts
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
        # Check if it's an HTML file
        if html_file.suffix in ['.html', '.htm'] or str(html_file).endswith('.html.gz'):
            html_files.append(html_file)

    if not html_files:
        return {'domain': domain, 'converted': 0, 'skipped': 0}

    # Prepare arguments for parallel processing
    file_args = [(f, domain_folder, output_path, filenames_to_remove) for f in html_files]

    converted = 0
    skipped = 0

    # Determine number of workers for files within this domain
    file_workers = max_workers if max_workers is not None else min(2, mp.cpu_count())

    # Process files in parallel using threads (I/O bound)
    try:
        with ThreadPoolExecutor(max_workers=file_workers) as executor:
            for status, count in executor.map(process_single_html_file, file_args):
                if status == 'converted':
                    converted += count
                else:
                    skipped += count
    except Exception as e:
        print(f"Error in thread pool for domain {domain}: {e}")
        # Mark remaining files as skipped
        skipped += len(file_args) - (converted + skipped)
    finally:
        # Clean up memory after processing this domain
        del html_files
        del file_args
        gc.collect()

    return {'domain': domain, 'converted': converted, 'skipped': skipped}


def convert_html_combined_to_markdown(
    input_dir,
    output_dir,
    excel_path=None,
    mappings_path=None,
    filenames_to_remove=[
        "impressum", "datenschutz", "kontakt", "robots",
        "imprint", "data-protection", "contact", "copyright",
    ],
    max_domain_workers=None,  # Deprecated - kept for compatibility
    max_file_workers=None
):
    """
    Convert HTML files from html_combined directory to markdown files.
    Creates one markdown file per HTML page, maintaining folder structure.
    Uses ThreadPoolExecutor for file-level parallelism (domains processed sequentially).

    Args:
        input_dir (str): Path to html_combined directory
        output_dir (str): Path where markdown files should be saved
        excel_path (str, optional): Path to Excel file with URL column to filter domains.
            If None, processes all domains.
        mappings_path (str, optional): Path to save domain->URL mappings JSON file
        filenames_to_remove (list): Keywords that if found in filename, file is skipped
        max_domain_workers (int, optional): DEPRECATED - Kept for compatibility, not used.
        max_file_workers (int, optional): Max parallel files per domain. If None, uses min(16, CPU count).

    Returns:
        dict: Summary with 'domains_processed', 'files_converted', 'files_skipped'
    """
    print("=" * 70)
    print("HTML to Markdown Converter (Parallelized)")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    if excel_path:
        print(f"Excel:  {excel_path}")
    print("=" * 70)

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
                    # Store only the base URL (protocol + domain), not the full path
                    domain_mappings[base_site] = get_base_url_from_url(url)

        # Save mappings
        mappings_path_obj = Path(mappings_path)
        mappings_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(domain_mappings, f, indent=2)
        print(f"Saved domain mappings to {mappings_path}")

        # Clear large DataFrame from memory
        del df
        del urls
        gc.collect()  # Force garbage collection

    # Process files
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Walk through all domain folders in html_combined
    domain_folders = [d for d in input_path.iterdir() if d.is_dir()]

    # Filter domains if allowed_domains is provided
    if allowed_domains is not None:
        domain_folders = [d for d in domain_folders if d.name in allowed_domains]

    print(f"\nProcessing {len(domain_folders)} domain folders in parallel...")
    print("=" * 70)

    # OPTIMIZATION 4: Batch directory creation - Only top-level
    # Pre-create only top-level output directories (subdirs created during processing)
    print("Pre-creating output directories...")
    for domain_folder in domain_folders:
        domain_output_dir = output_path / domain_folder.name
        domain_output_dir.mkdir(parents=True, exist_ok=True)
        # Subdirectories will be created on-demand during file writing

    # Determine number of workers
    # Use ThreadPoolExecutor only (no ProcessPoolExecutor) to avoid worker crashes
    if max_file_workers is None:
        max_file_workers = min(16, mp.cpu_count())  # Use more threads since no process overhead

    print(f"Processing domains sequentially with {max_file_workers} file workers per domain")
    print("  (Using ThreadPoolExecutor only to avoid process crashes)")

    # Clear allowed_domains and domain_mappings from memory
    del allowed_domains
    if 'domain_mappings' in locals():
        del domain_mappings
    gc.collect()  # Force garbage collection

    results = []
    failed_domains = []

    # Process domains sequentially to avoid ProcessPoolExecutor crashes
    for domain_folder in tqdm(domain_folders, desc="Processing domains", unit="domain"):
        domain_name = domain_folder.name
        try:
            result = process_domain_parallel(
                domain_folder=domain_folder,
                input_path=input_path,
                output_path=output_path,
                allowed_domains=None,
                filenames_to_remove=filenames_to_remove,
                max_workers=max_file_workers
            )
            results.append(result)
        except Exception as e:
            print(f"\nError processing domain {domain_name}: {e}")
            failed_domains.append(domain_name)
            # Add empty result to continue
            results.append({'domain': domain_name, 'converted': 0, 'skipped': 0})

    # Aggregate results
    domains_processed = {r['domain'] for r in results if r['converted'] > 0 or r['skipped'] > 0}
    files_converted = sum(r['converted'] for r in results)
    files_skipped = sum(r['skipped'] for r in results)

    print("\n" + "=" * 70)
    print(f"✓ Processed {len(domains_processed)} domains")
    print(f"✓ Converted {files_converted} files")
    print(f"✓ Skipped {files_skipped} files")
    if failed_domains:
        print(f"⚠ Failed {len(failed_domains)} domains: {', '.join(failed_domains[:5])}")
        if len(failed_domains) > 5:
            print(f"  ... and {len(failed_domains) - 5} more")
    print(f"✓ Output directory: {output_dir}")
    print("=" * 70)

    return {
        'domains_processed': len(domains_processed),
        'files_converted': files_converted,
        'files_skipped': files_skipped,
        'domains': sorted(list(domains_processed))
    }


def main():
    """Main function with default paths."""
    input_dir = "output/html_combined"
    output_dir = "output/markdown"
    excel_path = "data/2025-11-20_19945_topics.xlsx"
    mappings_path = "output/domain_mappings.json"

    convert_html_combined_to_markdown(
        input_dir,
        output_dir,
        excel_path=excel_path,
        mappings_path=mappings_path
    )


if __name__ == "__main__":
    main()

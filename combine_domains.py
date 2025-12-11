#!/usr/bin/env python3
"""
Combine HTML files from multiple WARC extractions by domain.
For files that exist in multiple folders (same domain, same path),
keep the one with the newest timestamp.
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import hashlib


def extract_timestamp_and_domain(folder_name):
    """
    Extract timestamp and domain from folder name.
    Format: {WARC_FILENAME}_{DOMAIN}
    Timestamp in WARC filename: YYYYMMDDHHMMSSmmm

    Example: ARCHIVEIT-19945-TEST-JOB2538000-0-SEED4432727-20250409125201867-00000-9618ziof.warc.gz_hytac.arch.ethz.ch
    Returns: (datetime_object, 'hytac.arch.ethz.ch')
    """
    # Split by last underscore to separate WARC filename from domain
    parts = folder_name.rsplit('_', 1)
    if len(parts) != 2:
        return None, None

    warc_filename, domain = parts

    # Extract timestamp from WARC filename (format: YYYYMMDDHHMMSS followed by milliseconds)
    # Look for pattern: 14 digits followed by optional digits
    timestamp_match = re.search(r'-(\d{14})\d*-', warc_filename)
    if not timestamp_match:
        return None, domain

    timestamp_str = timestamp_match.group(1)

    try:
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
        return timestamp, domain
    except ValueError:
        return None, domain


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


def scan_html_folders(input_dir, allowed_domains=None):
    """
    Scan all folders and organize by domain.

    Args:
        input_dir (str): Directory containing WARC extraction folders
        allowed_domains (set, optional): Set of allowed domains to filter by

    Returns: dict mapping domain -> list of (timestamp, folder_path) tuples
    """
    domain_folders = defaultdict(list)

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return domain_folders

    for folder in input_path.iterdir():
        if not folder.is_dir():
            continue

        timestamp, domain = extract_timestamp_and_domain(folder.name)
        if domain:
            # Filter by allowed domains if provided
            if allowed_domains is None or domain in allowed_domains:
                domain_folders[domain].append((timestamp, folder))
                print(f"Found: {domain} - {timestamp} - {folder.name}")
            else:
                print(f"Skipping (not in Excel): {domain} - {folder.name}")

    return domain_folders


def normalize_filename(filepath_str):
    """
    Normalize filenames by removing duplicate indicators like (1), (2), etc.

    Examples:
        'subfolder/en(1).html' -> 'subfolder/en.html'
        'subfolder/en(2).html' -> 'subfolder/en.html'
        'subfolder/en.html' -> 'subfolder/en.html'

    Returns:
        str: normalized path
    """
    path = Path(filepath_str)
    parent = path.parent

    # Pattern to match (N) before the extension
    pattern = r'\(\d+\)'

    # Split filename into stem and suffix
    stem = path.stem
    suffix = path.suffix

    # Remove all (N) patterns from the stem
    normalized_stem = re.sub(pattern, '', stem)

    # Reconstruct the normalized filename
    normalized_filename = f"{normalized_stem}{suffix}"
    normalized_path = str(parent / normalized_filename)

    return normalized_path


def get_file_hash_fast(file_path, chunk_size=8192):
    """
    Fast file hash using first chunk, middle chunk, and last chunk.
    This is much faster than hashing the entire file.
    """
    file_size = file_path.stat().st_size

    # For small files, just hash everything
    if file_size < chunk_size * 3:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    # For larger files: hash first chunk + middle chunk + last chunk
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        # First chunk
        hasher.update(f.read(chunk_size))

        # Middle chunk
        f.seek(file_size // 2)
        hasher.update(f.read(chunk_size))

        # Last chunk
        f.seek(-chunk_size, 2)
        hasher.update(f.read(chunk_size))

    return hasher.hexdigest()


def deduplicate_files(files_dict):
    """
    Deduplicate files with pattern like name.html, name(1).html, name(2).html.
    Uses file size first (fast), then content hash only when needed.

    Args:
        files_dict: dict mapping relative_path -> absolute_path

    Returns:
        dict: deduplicated dict mapping relative_path -> absolute_path
    """
    # Group files by their normalized name
    groups = defaultdict(list)

    for rel_path, abs_path in files_dict.items():
        normalized = normalize_filename(rel_path)
        groups[normalized].append((rel_path, abs_path))

    # For each group, pick the best file
    deduplicated = {}
    duplicates_removed = 0

    for normalized_path, file_list in groups.items():
        if len(file_list) == 1:
            # No duplicates, keep the file
            rel_path, abs_path = file_list[0]
            deduplicated[rel_path] = abs_path
        else:
            # Multiple files - deduplicate by size then hash
            duplicates_removed += len(file_list) - 1

            # Collect file info: (rel_path, abs_path, size)
            file_info = []
            for rel_path, abs_path in file_list:
                size = abs_path.stat().st_size
                file_info.append((rel_path, abs_path, size))

            # Sort by size descending (larger files are likely more complete)
            file_info.sort(key=lambda x: x[2], reverse=True)

            # Check if all files have the same size
            sizes = [info[2] for info in file_info]
            if len(set(sizes)) == 1:
                # Same size - check content hash
                hashes = {}
                for rel_path, abs_path, size in file_info:
                    file_hash = get_file_hash_fast(abs_path)
                    if file_hash not in hashes:
                        hashes[file_hash] = (rel_path, abs_path)

                # Pick first unique hash (prefer original name without (N))
                for rel_path, abs_path, _ in file_info:
                    file_hash = get_file_hash_fast(abs_path)
                    if hashes[file_hash][0] == rel_path:
                        deduplicated[rel_path] = abs_path
                        break
            else:
                # Different sizes - keep the largest one
                rel_path, abs_path, _ = file_info[0]
                deduplicated[rel_path] = abs_path

    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} duplicate files")

    return deduplicated


def get_all_files_in_folder(folder_path):
    """
    Get all HTML files in a folder with their relative paths.
    Returns: dict mapping relative_path -> absolute_path
    """
    files = {}
    folder_path = Path(folder_path)

    for file_path in folder_path.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(folder_path)
            files[str(relative_path)] = file_path

    return files


def combine_domain_folders(domain, folder_list, output_dir):
    """
    Combine multiple folders for the same domain.
    For duplicate files, keep the one with the newest timestamp.
    Deduplicates files with pattern like name.html, name(1).html, name(2).html.
    Returns file count and timestamp metadata.
    """
    print(f"\nProcessing domain: {domain}")
    print(f"  Found {len(folder_list)} folders")

    # Create output directory for this domain
    output_path = Path(output_dir) / domain
    output_path.mkdir(parents=True, exist_ok=True)

    # Track which file to use for each relative path
    file_registry = {}  # relative_path -> (timestamp, source_absolute_path)

    # Process each folder for this domain
    for timestamp, folder_path in folder_list:
        files = get_all_files_in_folder(folder_path)

        # Deduplicate files within this folder first (removes name(1).html, name(2).html, etc.)
        files = deduplicate_files(files)

        for relative_path, absolute_path in files.items():
            # If we haven't seen this file yet, or this version is newer
            if relative_path not in file_registry:
                file_registry[relative_path] = (timestamp, absolute_path)
                # print(f"  + {relative_path} (from {timestamp})")
            else:
                existing_timestamp, _ = file_registry[relative_path]

                # If timestamp is None, we can't compare, so keep the existing one
                if timestamp is None:
                    continue
                if existing_timestamp is None:
                    file_registry[relative_path] = (timestamp, absolute_path)
                    # print(f"  ↑ {relative_path} (updated to {timestamp})")
                elif timestamp > existing_timestamp:
                    file_registry[relative_path] = (timestamp, absolute_path)
                    # print(f"  ↑ {relative_path} (updated: {existing_timestamp} -> {timestamp})")

    # Copy all selected files to output directory and build metadata
    print(f"\n  Copying {len(file_registry)} files to {output_path}")
    timestamp_metadata = {}

    for relative_path, (timestamp, source_path) in file_registry.items():
        dest_path = output_path / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

        # Store timestamp metadata
        # Key: domain/relative_path, Value: ISO timestamp string
        file_key = f"{domain}/{relative_path}"
        timestamp_metadata[file_key] = timestamp.isoformat() if timestamp else None

    print(f"  ✓ Completed {domain}")
    return len(file_registry), timestamp_metadata


def _process_domain_worker(args):
    """
    Worker function for parallel domain processing.
    Accepts a tuple of (domain, folder_list, output_dir) and returns results.
    """
    domain, folder_list, output_dir = args
    # Sort by timestamp (None values go first)
    folder_list.sort(key=lambda x: x[0] if x[0] is not None else datetime.min)
    files_count, timestamp_metadata = combine_domain_folders(domain, folder_list, output_dir)
    return domain, files_count, timestamp_metadata


def combine_domains_by_timestamp(input_dir, output_dir, timestamps_json_path=None, excel_path=None, max_workers=None):
    """
    Combine HTML files from multiple WARC extractions by domain.
    For files that exist in multiple folders (same domain, same path),
    keep the one with the newest timestamp.

    Args:
        input_dir (str): Directory containing the extracted WARC folders (e.g., "output/html_raw")
        output_dir (str): Directory where combined results will be saved (e.g., "output/html_combined")
        timestamps_json_path (str, optional): Path to save file timestamp metadata JSON.
            If None, saves to "{output_dir}_timestamps.json"
        excel_path (str, optional): Path to Excel file with URL column to filter domains.
            If None, processes all domains.
        max_workers (int, optional): Maximum number of parallel workers. If None, uses min(cpu_count(), 8).

    Returns:
        dict: Summary with keys 'domains_count', 'total_files', 'domains', and 'timestamps_file'
    """
    print("=" * 70)
    print("Domain HTML Combiner")
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

    # Scan folders and organize by domain
    domain_folders = scan_html_folders(input_dir, allowed_domains)

    if not domain_folders:
        print("\nNo valid folders found!")
        return {
            'domains_count': 0,
            'total_files': 0,
            'domains': [],
            'timestamps_file': None
        }

    print(f"\n\nFound {len(domain_folders)} unique domains")

    # Determine worker count (avoid too many workers for I/O-bound tasks)
    if max_workers is None:
        max_workers = min(cpu_count(), 8)

    print(f"Using {max_workers} parallel workers")
    print("=" * 70)

    # Process domains in parallel
    total_files = 0
    processed_domains = []
    all_timestamps = {}

    # Prepare work items for parallel processing
    work_items = [(domain, folder_list, output_dir) for domain, folder_list in sorted(domain_folders.items())]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(_process_domain_worker, work_item): work_item[0]
                   for work_item in work_items}

        # Collect results as they complete
        for future in as_completed(futures):
            domain = futures[future]
            try:
                domain, files_count, timestamp_metadata = future.result()
                total_files += files_count
                processed_domains.append(domain)
                all_timestamps.update(timestamp_metadata)
            except Exception as e:
                print(f"Error processing domain {domain}: {e}")

    # Save timestamp metadata to JSON
    if timestamps_json_path is None:
        timestamps_json_path = f"{output_dir}_timestamps.json"

    timestamps_path = Path(timestamps_json_path)
    timestamps_path.parent.mkdir(parents=True, exist_ok=True)

    with open(timestamps_path, 'w', encoding='utf-8') as f:
        json.dump(all_timestamps, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"✓ Successfully combined {len(domain_folders)} domains")
    print(f"✓ Total files: {total_files}")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Timestamps saved to: {timestamps_json_path}")
    print("=" * 70)

    return {
        'domains_count': len(domain_folders),
        'total_files': total_files,
        'domains': processed_domains,
        'timestamps_file': str(timestamps_json_path)
    }


def main():
    """Main function to combine domains with default paths."""
    # Configuration
    input_dir = "output/html_raw/19945"
    output_dir = "output/html_combined"

    combine_domains_by_timestamp(input_dir, output_dir)


if __name__ == "__main__":
    main()

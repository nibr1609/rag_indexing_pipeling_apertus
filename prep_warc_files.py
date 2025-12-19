"""
Given warc files create md and text files.
using: https://github.com/recrm/ArchiveTools#warc-extractorpy
"""
import subprocess
import os
import json
import re
import gzip
import pandas as pd
from bs4 import BeautifulSoup
from html_to_markdown import convert_to_markdown
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def process_single_warc_file(warc_file, content_type, output_dir_path):
    """
    Process a single WARC file to extract HTML or PDF content.

    Args:
        warc_file (Path): Path to the WARC file
        content_type (str): Content type to extract ('text/html' or 'pdf')
        output_dir_path (str): Output directory path

    Returns:
        tuple: (warc_filename, success, error_message)
    """
    try:
        # warc_extractor.py expects the parent directory as -path and will process all WARC files in it
        # So we need to create a temporary directory structure or call it differently
        # For now, let's just call it on the parent directory with the specific file
        warc_file_str = str(warc_file)

        # Create a temporary symlink directory approach won't work well
        # Instead, call warc_extractor directly on the file by putting it in -path
        # The issue is warc_extractor expects a directory, so we pass the parent and let it find the file
        # But that would process all files again...

        # Better approach: Read the warc_extractor code - it treats -path as a directory
        # We need to work around this limitation
        # Let's create a temp directory with a symlink to the single file
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create symlink to the warc file in temp directory
            temp_link = Path(temp_dir) / warc_file.name
            temp_link.symlink_to(warc_file.absolute())

            cmd = f"python warc_extractor.py http:content-type:{content_type} -dump content -error -path {temp_dir} -output_path {output_dir_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                return (warc_file.name, True, None)
            else:
                return (warc_file.name, False, result.stderr)
    except Exception as e:
        return (warc_file.name, False, str(e))


def warc_to_html(input_dir_path: str, output_dir_path: str, max_workers=None):
    """
    Goes through the files in `input_dir_path`, finds all the warc (and warc.gz) files,
    extracts the html pages and saves them in the given `output_dir_path`.
    The hierarchy of directories is preserved for the html output.
    Processes WARC files in parallel for better performance.

    Args:
        input_dir_path (str): Path to the input directory.
        output_dir_path (str): Path to the output directory.
        max_workers (int, optional): Maximum number of parallel workers. If None, uses CPU count.
    """
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Find all WARC files
    input_path = Path(input_dir_path)
    warc_files = list(input_path.glob("*.warc")) + list(input_path.glob("*.warc.gz"))

    if not warc_files:
        print(f"No WARC files found in {input_dir_path}")
        return

    # Determine number of workers
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(warc_files))

    print(f"Processing {len(warc_files)} WARC files for HTML extraction using {max_workers} workers...")

    # Process WARC files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_warc_file, warc_file, "text/html", output_dir_path)
            for warc_file in warc_files
        ]

        successful = 0
        failed = 0

        for future in tqdm(as_completed(futures), total=len(warc_files), desc="Extracting HTML", unit="file"):
            filename, success, error = future.result()
            if success:
                successful += 1
            else:
                failed += 1
                if error:
                    print(f"\nError processing {filename}: {error}")

    print(f"HTML extraction complete: {successful} successful, {failed} failed")


def warc_to_pdf(input_dir_path: str, output_dir_path: str, max_workers=None):
    """
    Goes through the files in `input_dir_path`, finds all the warc (and warc.gz) files,
    extracts the pdf files and saves them in a `output_dir_path/wp-content` folder.
    Processes WARC files in parallel for better performance.

    Args:
        input_dir_path (str): Path to the input directory.
        output_dir_path (str): Path to the output directory.
        max_workers (int, optional): Maximum number of parallel workers. If None, uses CPU count.
    """
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Find all WARC files
    input_path = Path(input_dir_path)
    warc_files = list(input_path.glob("*.warc")) + list(input_path.glob("*.warc.gz"))

    if not warc_files:
        print(f"No WARC files found in {input_dir_path}")
        return

    # Determine number of workers
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(warc_files))

    print(f"Processing {len(warc_files)} WARC files for PDF extraction using {max_workers} workers...")

    # Process WARC files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_warc_file, warc_file, "pdf", output_dir_path)
            for warc_file in warc_files
        ]

        successful = 0
        failed = 0

        for future in tqdm(as_completed(futures), total=len(warc_files), desc="Extracting PDF", unit="file"):
            filename, success, error = future.result()
            if success:
                successful += 1
            else:
                failed += 1
                if error:
                    print(f"\nError processing {filename}: {error}")

    print(f"PDF extraction complete: {successful} successful, {failed} failed")
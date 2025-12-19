#!/usr/bin/env python3
"""
High-Performance PDF to Markdown Converter.
Uses ProcessPoolExecutor (Multiproc) with a flattened task list to bypass the GIL.
Integrates domain filtering, file exclusion, and mapping generation.
"""

import os
import json
import gzip
import re
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import time
import gc

# ==========================================
# Helper Functions
# ==========================================

def get_base_site_from_url(url_in):
    if "//" not in url_in:
        base_site = url_in
    else:
        url_in_old = url_in
        base_site = url_in.split("//")[1]
        if base_site == "http:":
            base_site = url_in_old.split("//")[2]

    base_site = base_site.replace("dns:", "").replace("mailto:", "")
    base_site = re.sub(r'^www\d*\.', '', base_site) # Remove www, www1, etc.
    base_site = base_site.split(":")[0]
    base_site = base_site.split("/")[0]

    if base_site.endswith("."):
        base_site = base_site[:-1]

    return base_site

def get_base_url_from_url(url_in):
    if "//" not in url_in:
        return url_in
    parts = url_in.split("//")
    return parts[0] + "//" + parts[1].split("/")[0]

def load_allowed_domains(excel_path):
    print(f"Loading allowed domains from: {excel_path}")
    df = pd.read_excel(excel_path)
    df = df.fillna("")
    urls = list(df["URL"])
    
    allowed = set()
    for url in urls:
        if url:
            allowed.add(get_base_site_from_url(url))
    
    print(f"Loaded {len(allowed)} allowed domains.")
    return allowed

def generate_domain_mappings(excel_path, mappings_path):
    print("Generating domain mappings...")
    df = pd.read_excel(excel_path)
    df = df.fillna("")
    
    domain_mappings = {}
    for url in df["URL"]:
        if url:
            base_site = get_base_site_from_url(url)
            if base_site not in domain_mappings:
                domain_mappings[base_site] = get_base_url_from_url(url)
    
    out_path = Path(mappings_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(domain_mappings, f, indent=2)
    print(f"Saved mappings to {mappings_path}")

# ==========================================
# The Worker Function (Must be top-level)
# ==========================================

def convert_single_pdf_task(task):
    """
    Worker function. Returns (status, full_path_string, reason).
    """
    input_path, output_path = task
    
    # Store full path as string for reporting
    full_path_str = str(input_path)
    
    try:
        import fitz 
        
        # Use your silencer context manager here if you have it
        # with suppress_c_stderr():
        if True:
            # 1. Open PDF
            if str(input_path).endswith('.gz'):
                with gzip.open(input_path, 'rb') as f:
                    pdf_bytes = f.read()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            else:
                doc = fitz.open(str(input_path))
                
            if doc.page_count == 0:
                doc.close()
                return ('skipped', full_path_str, "Empty Page Count")

            # 2. Extract Text
            text_parts = []
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"## Page {i+1}\n\n{text}")
            doc.close()
            
            full_text = "\n".join(text_parts)

            # 3. Clean Text
            full_text = re.sub(r"¬\n", "", full_text)
            full_text = re.sub(r'\\', "", full_text)
            full_text = full_text.strip()

            if not full_text:
                return ('skipped', full_path_str, "No Text Extracted (Scanned PDF?)")

            # 4. Write to Disk
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# PDF Document: {input_path.name}\n\n")
                f.write(full_text)
                
            return ('converted', full_path_str, None)

    except Exception as e:
        return ('failed', full_path_str, str(e))
# ==========================================
# Main Processing Logic
# ==========================================

def convert_pdf_combined_to_markdown(
    input_dir,
    output_dir,
    excel_path=None,
    mappings_path=None,
    filenames_to_remove=None
):
    print("=" * 60)
    print("High-Performance PDF Pipeline (Flattened Process Pool)")
    print("=" * 60)
    
    start_total = time.time()
    input_path_obj = Path(input_dir)
    output_path_obj = Path(output_dir)
    
    if filenames_to_remove is None:
        filenames_to_remove = []

    # 1. Handle Excel Logic (Filtering & Mappings)
    allowed_domains = None
    if excel_path:
        allowed_domains = load_allowed_domains(excel_path)
        if mappings_path:
            generate_domain_mappings(excel_path, mappings_path)

    # 2. Collect Tasks
    print("\nScanning and collecting files...")
    tasks = []
    
    domain_folders = [d for d in input_path_obj.iterdir() if d.is_dir()]
    
    for domain_folder in domain_folders:
        if allowed_domains is not None and domain_folder.name not in allowed_domains:
            continue
            
        for root, _, files in os.walk(domain_folder):
            for file in files:
                if not (file.endswith('.pdf') or file.endswith('.pdf.gz')):
                    continue
                if any(k in file.lower() for k in filenames_to_remove):
                    continue

                src_file = Path(root) / file
                rel_path = src_file.relative_to(input_path_obj)
                
                if file.endswith('.pdf.gz'):
                    dest_name = file.replace('.pdf.gz', '.md')
                else:
                    dest_name = Path(file).stem + '.md'
                
                dest_file = output_path_obj / rel_path.parent / dest_name
                tasks.append((src_file, dest_file))

    print(f"Found {len(tasks)} valid PDF files to process.")

    # 3. Parallel Execution
    cpu_count = mp.cpu_count()
    workers = max(1, cpu_count - 1) if cpu_count > 4 else cpu_count
    
    print(f"Starting ProcessPoolExecutor with {workers} workers...")
    
    results_summary = {'converted': 0, 'skipped': 0, 'failed': 0}
    skipped_examples = []
    failed_examples = []
    
    chunk_size = 20 

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(convert_single_pdf_task, tasks, chunksize=chunk_size)
        
        # Unpack the tuple: (status, full_path, reason)
        for status, full_path, reason in tqdm(futures, total=len(tasks), unit="pdf"):
            results_summary[status] += 1
            
            if status == 'skipped':
                # Only keep first 20 examples
                if len(skipped_examples) < 20:
                    skipped_examples.append(f"{full_path}  [{reason}]")
            
            elif status == 'failed':
                # Only keep first 20 examples
                if len(failed_examples) < 20:
                    failed_examples.append(f"{full_path}  [{reason}]")

    duration = time.time() - start_total
    rate = len(tasks) / duration if duration > 0 else 0

    print("\n" + "=" * 60)
    print(f"DONE in {duration:.2f}s ({rate:.1f} files/s)")
    print(f"✓ Converted: {results_summary['converted']}")
    print(f"✓ Skipped:   {results_summary['skipped']}")
    print(f"⚠ Failed:    {results_summary['failed']}")
    print("-" * 60)
    
    if skipped_examples:
        print("SAMPLE SKIPPED FILES (Full Paths):")
        for s in skipped_examples:
            print(f"  - {s}")
        if results_summary['skipped'] > 20:
            print(f"  ... and {results_summary['skipped'] - 20} more.")
            
    if failed_examples:
        print("\nSAMPLE FAILED FILES (Full Paths):")
        for f in failed_examples:
            print(f"  - {f}")
        if results_summary['failed'] > 20:
            print(f"  ... and {results_summary['failed'] - 20} more.")
            
    print("=" * 60)

# ==========================================
# Call Structure
# ==========================================

def process_pdf_pipeline():
    # Define paths
    pdf_raw_dir = "output/pdf_raw" # Example path
    pdf_combined_dir = "output/pdf_combined/19945"
    markdown_output_dir = "output/markdown_pdf/19945"
    topics_excel_path = "data/2025-11-20_19945_topics.xlsx"
    timestamps_json_path = "output/timestamps.json"
    pdf_mappings_path = "output/mappings/19945/pdf_domain_mappings.json"

    # Step 1: Combine (Commented out)
    # combine_domains_by_timestamp(
    #     input_dir=pdf_raw_dir,
    #     output_dir=pdf_combined_dir,
    #     timestamps_json_path=timestamps_json_path,
    #     excel_path=topics_excel_path
    # )

    # Step 2: Convert
    exclude_list = [
        "impressum", "datenschutz", "kontakt", "robots",
        "imprint", "data-protection", "contact", "copyright",
        "5_bafu_2009_bewertung_n_l_"  # The killer file
    ]

    convert_pdf_combined_to_markdown(
        input_dir=pdf_combined_dir,
        output_dir=markdown_output_dir,
        excel_path=topics_excel_path,
        mappings_path=pdf_mappings_path,
        filenames_to_remove=exclude_list
    )

if __name__ == "__main__":
    print("Doing pdf pipeline")
    process_pdf_pipeline()
#!/usr/bin/env python3
"""
Convert PDF files from pdf_combined directory to markdown files.
Filters domains based on Excel sheet and maintains folder structure.
Creates one markdown file per PDF.
"""

import os
import json
import gzip
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd


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


def get_base_url_from_url(url_in):
    """
    Extracts the base URL (protocol + domain) from a full URL.
    """
    if "//" not in url_in:
        return url_in

    parts = url_in.split("//")
    protocol = parts[0] + "//"
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


def convert_pdf_to_markdown(pdf_path, output_path):
    """
    Convert a single PDF file to markdown.
    
    Args:
        pdf_path (Path): Path to PDF file
        output_path (Path): Path where markdown should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import fitz  # PyMuPDF
        
        # Read PDF file (handle both regular and gzipped)
        if str(pdf_path).endswith('.gz'):
            with gzip.open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        else:
            pdf_doc = fitz.open(str(pdf_path))

        if pdf_doc.page_count == 0:
            pdf_doc.close()
            return False

        # Extract text from all pages
        text_parts = []
        for page_num, page in enumerate(pdf_doc):
            page_text = page.get_text()
            if page_text.strip():
                # Add page header in markdown
                text_parts.append(f"## Page {page_num + 1}\n\n{page_text}\n")

        pdf_doc.close()

        # Combine all text
        pdf_text = "\n".join(text_parts)

        # Clean up text
        pdf_text = re.sub("¬\n", "", pdf_text)  # Remove soft hyphens
        pdf_text = re.sub(r'\\', "", pdf_text)  # Remove backslashes
        pdf_text = pdf_text.strip()

        # Skip empty files
        if not pdf_text or pdf_text == "":
            return False

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write markdown file
        with open(output_path, 'w', encoding='utf-8') as f:
            # Add metadata header
            f.write(f"# PDF Document: {pdf_path.name}\n\n")
            f.write(pdf_text)

        return True

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return False


def convert_pdf_combined_to_markdown(
    input_dir,
    output_dir,
    excel_path=None,
    mappings_path=None,
    filenames_to_remove=[
        "impressum", "datenschutz", "kontakt", "robots",
        "imprint", "data-protection", "contact", "copyright",
    ]
):
    """
    Convert PDF files from pdf_combined directory to markdown files.
    Creates one markdown file per PDF, maintaining folder structure.

    Args:
        input_dir (str): Path to pdf_combined directory
        output_dir (str): Path where markdown files should be saved
        excel_path (str, optional): Path to Excel file with URL column to filter domains
        mappings_path (str, optional): Path to save domain->URL mappings JSON file
        filenames_to_remove (list): Keywords that if found in filename, file is skipped

    Returns:
        dict: Summary with 'domains_processed', 'files_converted', 'files_skipped'
    """
    print("=" * 70)
    print("PDF to Markdown Converter")
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
                    domain_mappings[base_site] = get_base_url_from_url(url)

        # Save mappings
        mappings_path_obj = Path(mappings_path)
        mappings_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(domain_mappings, f, indent=2)
        print(f"Saved domain mappings to {mappings_path}")

    # Process files
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    files_converted = 0
    files_skipped = 0
    domains_processed = set()

    # Walk through all domain folders in pdf_combined
    domain_folders = [d for d in input_path.iterdir() if d.is_dir()]

    print(f"\nProcessing {len(domain_folders)} domain folders...")
    print("=" * 70)

    for domain_folder in tqdm(domain_folders, desc="Processing domains", unit="domain"):
        domain = domain_folder.name

        # Check if domain is in allowed list (if filtering is enabled)
        if allowed_domains is not None and domain not in allowed_domains:
            continue

        domains_processed.add(domain)

        # Walk through all PDF files in this domain folder
        for pdf_file in domain_folder.rglob('*'):
            if not pdf_file.is_file():
                continue

            # Check if it's a PDF file
            if not (pdf_file.suffix == '.pdf' or str(pdf_file).endswith('.pdf.gz')):
                continue

            # Check if filename should be skipped
            skip_file = False
            for keyword in filenames_to_remove:
                if keyword in pdf_file.name.lower():
                    skip_file = True
                    files_skipped += 1
                    break

            if skip_file:
                continue

            # Calculate relative path from domain folder
            rel_path = pdf_file.relative_to(domain_folder)

            # Create output path (change extension to .md)
            if str(pdf_file).endswith('.pdf.gz'):
                md_filename = pdf_file.name.replace('.pdf.gz', '.md')
            else:
                md_filename = pdf_file.stem + '.md'

            output_file_path = output_path / domain / rel_path.parent / md_filename

            # Convert PDF to markdown
            if convert_pdf_to_markdown(pdf_file, output_file_path):
                files_converted += 1
            else:
                files_skipped += 1

    print("\n" + "=" * 70)
    print(f"✓ Processed {len(domains_processed)} domains")
    print(f"✓ Converted {files_converted} files")
    print(f"✓ Skipped {files_skipped} files")
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
    input_dir = "output/pdf_combined/19945"
    output_dir = "output/markdown_pdf/19945"
    excel_path = "data/2025-11-20_19945_topics.xlsx"
    mappings_path = "output/mappings/19945/pdf_domain_mappings.json"

    convert_pdf_combined_to_markdown(
        input_dir,
        output_dir,
        excel_path=excel_path,
        mappings_path=mappings_path
    )


if __name__ == "__main__":
    main()
import argparse
from prep_warc_files import warc_to_html, warc_to_pdf
import os
from combine_domains import combine_domains_by_timestamp
from html_combined_to_markdown import convert_html_combined_to_markdown
from pdf_combined_to_markdown import convert_pdf_combined_to_markdown
from index_to_elasticsearch import index_markdown_to_elasticsearch
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

def main():
    coll = "19945"
    parser = argparse.ArgumentParser(
        description="Process WARC files into Markdown and index to Elasticsearch.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--warc-input-dir',
        required=True,
        help="Path to the directory containing WARC files (e.g., './data/ethz_websites_2022-2025_examples')."
    )

    parser.add_argument(
        '--topics-excel-path',
        required=True
    )

    parser.add_argument(
        '--indexed-files-path',
        required=False,
        default=None,
        help="Optional path to JSON file tracking already indexed files for incremental indexing."
    )

    args = parser.parse_args()
    warc_input_dir = args.warc_input_dir
    topics_excel_path = args.topics_excel_path
    indexed_files_path_arg = args.indexed_files_path

    # 4. Read Elasticsearch configuration from .env file
    es_username = os.getenv('ELASTIC_USERNAME')
    es_password = os.getenv('ELASTIC_PASSWORD')
    es_url = os.getenv('ES_URL', 'https://es.swissai.cscs.ch')
    embedding_service_url = os.getenv('EMBEDDING_SERVICE_URL')
    index_name = os.getenv('INDEX_NAME', 'ethz_webarchive')

    if not es_username:
        raise ValueError("ELASTIC_USERNAME not found in environment variables. Please add it to your .env file")
    if not es_password:
        raise ValueError("ELASTIC_PASSWORD not found in environment variables. Please add it to your .env file")
    if not embedding_service_url:
        raise ValueError("EMBEDDING_SERVICE_URL not found in environment variables. Please add it to your .env file")

    # Define dynamic output paths based on collection ID
    output_base_dir = "output"
    html_raw_dir = os.path.join(output_base_dir, "html_raw", coll)
    pdf_raw_dir = os.path.join(output_base_dir, "pdf_raw", coll)
    html_combined_dir = os.path.join(output_base_dir, "html_combined", coll)
    pdf_combined_dir = os.path.join(output_base_dir, "pdf_combined", coll)
    markdown_output_dir = os.path.join(output_base_dir, "markdown", coll)
    mappings_base_dir = os.path.join(output_base_dir, "mappings", coll)

    timestamps_json_path = os.path.join(mappings_base_dir, "timestamps.json")
    html_mappings_path = os.path.join(mappings_base_dir, "domain_mappings.json")
    pdf_mappings_path = os.path.join(mappings_base_dir, "pdf_domain_mappings.json")

    # Use provided indexed files path or default
    if indexed_files_path_arg:
        indexed_files_path = indexed_files_path_arg
    else:
        indexed_files_path = os.path.join(output_base_dir, f"{index_name}_indexed_files.json")

    # Ensure necessary directories exist
    os.makedirs(mappings_base_dir, exist_ok=True)
    os.makedirs(markdown_output_dir, exist_ok=True)



    # -------------------
    # Processing Pipeline (Parallelized)
    # -------------------

    print("\n" + "=" * 70)
    print("STAGE 1: WARC Extraction (HTML and PDF in parallel)")
    print("=" * 70)

    # OPTIMIZATION 2: Run HTML and PDF WARC extraction in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        html_future = executor.submit(warc_to_html, warc_input_dir, html_raw_dir)
        pdf_future = executor.submit(warc_to_pdf, warc_input_dir, pdf_raw_dir)

        # Wait for both to complete
        html_future.result()
        pdf_future.result()

    print("\n" + "=" * 70)
    print("STAGE 2: Domain Combination and Markdown Conversion (HTML and PDF in parallel)")
    print("=" * 70)

    # OPTIMIZATION 2: Run HTML and PDF processing pipelines in parallel
    def process_html_pipeline():
        combine_domains_by_timestamp(
            input_dir=html_raw_dir,
            output_dir=html_combined_dir,
            timestamps_json_path=timestamps_json_path,
            excel_path=topics_excel_path
        )
        convert_html_combined_to_markdown(
            input_dir=html_combined_dir,
            output_dir=markdown_output_dir,
            excel_path=topics_excel_path,
            mappings_path=html_mappings_path
        )

    def process_pdf_pipeline():
        combine_domains_by_timestamp(
            input_dir=pdf_raw_dir,
            output_dir=pdf_combined_dir,
            timestamps_json_path=timestamps_json_path,
            excel_path=topics_excel_path
        )
        convert_pdf_combined_to_markdown(
            input_dir=pdf_combined_dir,
            output_dir=markdown_output_dir,
            excel_path=topics_excel_path,
            mappings_path=pdf_mappings_path
        )

    # Run HTML and PDF pipelines concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        html_pipeline_future = executor.submit(process_html_pipeline)
        pdf_pipeline_future = executor.submit(process_pdf_pipeline)

        # Wait for both pipelines to complete
        html_pipeline_future.result()
        pdf_pipeline_future.result()

    print("\n" + "=" * 70)
    print("STAGE 3: Elasticsearch Indexing")
    print("=" * 70)

    ethz = os.path.join(output_base_dir, "markdown", coll, "ethz.ch")


    index_markdown_to_elasticsearch(
        clean_index=False,
        es_user=es_username,
        es_password=es_password,
        es_url=es_url,
        markdown_dir=ethz,
        index_name=index_name,
        mappings_path=html_mappings_path,
        timestamps_path=timestamps_json_path,
        force_domain="ethz.ch",
        indexed_files_path=indexed_files_path
    )


if __name__ == "__main__":
    main()
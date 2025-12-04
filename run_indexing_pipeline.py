from prep_warc_files import warc_to_html, warc_to_pdf
import os
from combine_domains import combine_domains_by_timestamp
from html_combined_to_markdown import convert_html_combined_to_markdown
from pdf_combined_to_markdown import convert_pdf_combined_to_markdown
from index_to_elasticsearch import index_markdown_to_elasticsearch
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read Elasticsearch configuration from .env file
es_username = os.getenv('ELASTIC_USERNAME')
es_password = os.getenv('ELASTIC_PASSWORD')
es_url = os.getenv('ES_URL', 'https://es.swissai.cscs.ch')
embedding_model = os.getenv('EMBEDDING_MODEL', 'all-minilm')
index_name = os.getenv('INDEX_NAME', 'ethz_webarchive')

if not es_username:
    raise ValueError("ELASTIC_USERNAME not found in environment variables. Please add it to your .env file")
if not es_password:
    raise ValueError("ELASTIC_PASSWORD not found in environment variables. Please add it to your .env file") 

if os.path.exists("output"):
    shutil.rmtree("output")

coll_list = ["19945"]

for coll in coll_list:
    warc_to_html("./data/ethz_websites_2022-2025_examples", "output/html_raw/"+coll+"/")
    warc_to_pdf("./data/ethz_websites_2022-2025_examples", "output/pdf_raw/"+coll+"/")

    result = combine_domains_by_timestamp(
        input_dir="output/html_raw/"+coll,
        output_dir="output/html_combined/"+coll,
        timestamps_json_path="output/mappings/"+coll+"/timestamps.json"
    )

    print(f"Processed {result['domains_count']} domains")
    print(f"Total files: {result['total_files']}")
    print(f"Domains: {result['domains']}")


    result = convert_html_combined_to_markdown(
        input_dir="output/html_combined/"+coll,
        output_dir="output/markdown/"+coll,
        excel_path="data/2025-11-20_19945_topics.xlsx",
        mappings_path="output/mappings/"+coll+"/domain_mappings.json"
    )

    result = combine_domains_by_timestamp(
        input_dir="output/pdf_raw/"+coll,
        output_dir="output/pdf_combined/"+coll,
        timestamps_json_path="output/mappings/"+coll+"/timestamps.json"
    )

    print(f"Processed {result['domains_count']} domains")
    print(f"Total files: {result['total_files']}")
    print(f"Domains: {result['domains']}")

    result = convert_pdf_combined_to_markdown(
        input_dir="output/pdf_combined/"+coll,
        output_dir="output/markdown/"+coll,  # Same output dir as HTML markdown
        excel_path="data/2025-11-20_19945_topics.xlsx",
        mappings_path="output/mappings/"+coll+"/pdf_domain_mappings.json"
    )

    index_markdown_to_elasticsearch(
        clean_index=True,
        es_user=es_username,
        es_password=es_password,
        es_url=es_url,
        embedding_model=embedding_model,
        markdown_dir="output/markdown/"+coll,
        index_name=index_name,
        mappings_path="output/mappings/"+coll+"/domain_mappings.json",
        timestamps_path="output/mappings/"+coll+"/timestamps.json"
    )
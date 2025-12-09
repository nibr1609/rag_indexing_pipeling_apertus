# Use standard Python 3.11 from Docker Hub (Public, no login needed)
FROM python:3.11-slim

# 1. Install ESSENTIAL system tools needed for compiling Python packages
# This list is focused on non-GUI, core dependencies for data science and networking.
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    libsqlite3-dev \
    libbz2-dev \
    libssl-dev \
    libffi-dev \
    liblzma-dev \
    libicu-dev \
    libexpat1-dev \
    zlib1g-dev \
    # Remove lists cache to minimize final image size
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip
RUN pip install --upgrade pip

# 3. Install your Python dependencies (using a simplified, single block)
# NOTE: The list is the same as before, just included for completeness.
RUN pip install \
    Deprecated==1.2.18 Jinja2==3.1.6 MarkupSafe==3.0.3 PyMuPDF==1.26.6 PyYAML==6.0.3 \
    SQLAlchemy==2.0.44 aiohappyeyeballs==2.6.1 aiohttp==3.13.2 aiosignal==1.4.0 \
    aiosqlite==0.21.0 annotated-types==0.7.0 anyio==4.12.0 attrs==25.4.0 banks==2.2.0 \
    beautifulsoup4==4.14.3 certifi==2025.11.12 charset-normalizer==3.4.4 click==8.3.1 \
    colorama==0.4.6 dataclasses-json==0.6.7 defusedxml==0.7.1 dirtyjson==1.0.8 \
    distro==1.9.0 elastic-transport==8.17.1 elasticsearch==8.13.2 et_xmlfile==2.0.0 \
    filelock==3.20.0 filetype==1.2.0 frozenlist==1.8.0 fsspec==2025.12.0 greenlet==3.2.4 \
    griffe==1.15.0 h11==0.16.0 hf-xet==1.2.0 html-to-markdown==2.9.2 httpcore==1.0.9 \
    httpx==0.28.1 huggingface-hub==0.36.0 idna==3.11 iniconfig==2.3.0 jiter==0.12.0 \
    joblib==1.5.2 llama-cloud==0.1.35 llama-cloud-services==0.6.54 llama-index==0.14.9 \
    llama-index-cli==0.5.3 llama-index-core==0.14.9 llama-index-embeddings-ollama==0.8.4 \
    llama-index-embeddings-openai==0.5.1 llama-index-indices-managed-llama-cloud==0.9.4 \
    llama-index-instrumentation==0.4.2 llama-index-llms-ollama==0.9.0 \
    llama-index-llms-openai==0.6.10 llama-index-readers-file==0.5.5 \
    llama-index-readers-llama-parse==0.5.1 llama-index-vector-stores-elasticsearch==0.5.1 \
    llama-index-workflows==2.11.5 llama-parse==0.6.54 marshmallow==3.26.1 mpmath==1.3.0 \
    multidict==6.7.0 mypy_extensions==1.1.0 networkx==3.6 nltk==3.9.2 numpy==2.3.5 \
    ollama==0.6.1 openai==2.8.1 openpyxl==3.1.5 pandas==2.2.3 pillow==12.0.0 pluggy==1.6.0 \
    propcache==0.4.1 pydantic==2.12.5 pydantic_core==2.41.5 pypdf==6.4.0 pytest==9.0.1 \
    pytest-asyncio==1.3.0 python-dotenv==1.2.1 pytz==2025.2 regex==2025.11.3 \
    requests==2.32.5 safetensors==0.7.0 scikit-learn==1.7.2 scipy==1.16.3 \
    sentence-transformers==5.1.2 sniffio==1.3.1 soupsieve==2.8 striprtf==0.0.26 \
    sympy==1.14.0 tenacity==9.1.2 threadpoolctl==3.6.0 tiktoken==0.12.0 \
    tokenizers==0.22.1 torch==2.2.2 tqdm==4.67.1 transformers==4.57.3 \
    typing-inspect==0.9.0 typing-inspection==0.4.2 tzdata==2025.2 urllib3==2.5.0 \
    wrapt==1.17.3 yarl==1.22.0

# 4. Set up workspace
RUN mkdir -p /workspace
WORKDIR /workspace

# 5. Default command
CMD ["/bin/bash"]
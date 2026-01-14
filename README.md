# RAG Pipeline with Docling and BGE-M3

A robust local RAG (Retrieval-Augmented Generation) pipeline that handles scanned PDFs and complex tables without data loss.

## 🎯 Why This Project?

Traditional PDF extraction tools often fail with:
- **Scanned documents**: Text extraction returns empty or garbled results
- **Complex tables**: Table structure is lost, data becomes unusable
- **Multi-column layouts**: Content gets jumbled or out of order
- **Images with text**: Embedded text in images is completely missed

**This project solves these problems** by combining:
- **IBM Docling**: A vision-based document understanding library that "sees" the document like a human would
- **BGE-M3 Embeddings**: State-of-the-art multilingual embeddings with excellent semantic understanding
- **ChromaDB**: Fast, local vector database that requires no external services

### Use Cases

- 📊 **Financial Document Analysis**: Extract tables from annual reports, balance sheets, income statements
- 📋 **Medical Records Processing**: Handle scanned health records with OCR
- 📑 **Legal Document Review**: Search across contracts and legal documents
- 🏢 **Enterprise Knowledge Base**: Build searchable archives of company documents
- 🔬 **Research Paper Analysis**: Query academic papers and extract findings

## ✨ Features

- **Docling (IBM)**: Vision-based layout analyzer for superior PDF extraction
  - Handles scanned PDFs with OCR
  - Extracts complex tables with structure preservation
  - Converts documents to clean Markdown format
  
- **BGE-M3 Embeddings**: State-of-the-art multilingual embeddings
  - 1024-dimensional dense embeddings
  - Supports sparse embeddings for hybrid search
  - Handles documents up to 8192 tokens
  
- **ChromaDB**: Local vector storage with persistence
  - Cosine similarity search
  - Metadata filtering
  - Easy to scale

## 📁 Project Structure

```
RAG-PDF-docling/
├── data/
│   ├── source_pdfs/           # 📁 Place your PDF files here
│   ├── processed_markdown/    # 📄 Extracted Markdown output
│   └── vector_store/          # 🗃️ ChromaDB vector database
├── src/
│   ├── __init__.py
│   ├── document_processor.py  # PDF extraction with Docling
│   ├── text_chunker.py        # Intelligent text chunking
│   ├── embedding_generator.py # BGE-M3 embedding generation
│   ├── vector_store.py        # ChromaDB wrapper
│   ├── retriever.py           # Document retrieval
│   └── rag_pipeline.py        # Main orchestrator
├── example_usage.py           # Example script to get started
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 📋 Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.11+ | 3.11 or 3.12 |
| **RAM** | 8 GB | 16 GB+ |
| **Disk Space** | 5 GB | 10 GB+ (for models) |
| **GPU** | Optional | NVIDIA with 8GB+ VRAM |

### Software Dependencies

1. **Python 3.11 or higher**
   ```bash
   python --version  # Should show 3.11.x or higher
   ```

2. **Git** (for cloning the repository)
   ```bash
   git --version
   ```

3. **pip** (Python package manager)
   ```bash
   pip --version
   ```

### Hardware Recommendations

- **With GPU (NVIDIA)**: Processing is 5-10x faster
  - CUDA 11.8 or 12.1 required
  - Minimum 8GB VRAM (RTX 3070 or better)
  
- **Without GPU**: Works on CPU, but slower
  - Expect 2-5 minutes per PDF page
  - Recommended for small document sets (<50 pages total)

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sreeram68/RAG-PDF-docling.git
cd RAG-PDF-docling
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

**Basic installation (CPU):**
```bash
pip install -r requirements.txt
```

**With GPU support (NVIDIA CUDA 12.1):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**With GPU support (NVIDIA CUDA 11.8):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "from src.rag_pipeline import RAGPipeline; print('✅ Installation successful!')"
```

## 📖 Usage

### 1. Add Your Documents

Place your PDF files in the `data/source_pdfs/` directory:

```bash
# Windows
copy "C:\path\to\your\document.pdf" data\source_pdfs\

# Linux/macOS
cp /path/to/your/document.pdf data/source_pdfs/
```

### 2. Ingest Documents

Process all PDFs and build the vector database:

```bash
python -m src.rag_pipeline --ingest
```

This will:
1. ✅ Process all PDFs using Docling (OCR + table extraction)
2. ✅ Convert to Markdown (saved in `data/processed_markdown/`)
3. ✅ Chunk text with semantic boundaries
4. ✅ Generate BGE-M3 embeddings
5. ✅ Store in ChromaDB vector database

### 3. Query Your Documents

```bash
python -m src.rag_pipeline --query "What is the quarterly revenue?"
```

### 4. Check Statistics

```bash
python -m src.rag_pipeline --stats
```

### 5. Run Example Script

```bash
python example_usage.py
```

## 🐍 Python API

```python
from src.rag_pipeline import RAGPipeline

# Initialize the pipeline
pipeline = RAGPipeline(
    source_dir="data/source_pdfs",
    output_dir="data/processed_markdown",
    vector_store_dir="data/vector_store",
    chunk_size=1000,
    chunk_overlap=200,
    top_k=5
)

# Ingest documents
stats = pipeline.ingest_documents()
print(f"Processed {stats['pdfs_processed']} PDFs")

# Query the system
result = pipeline.query("What are the key findings?")
for doc in result.results:
    print(f"Score: {doc.score:.3f} - {doc.content[:100]}...")

# Get formatted context for LLM
context = pipeline.get_context_for_llm("Summarize the main points")
print(context)
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. `ModuleNotFoundError: No module named 'docling'`

**Cause**: Docling is not installed or virtual environment is not activated.

**Solution**:
```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Reinstall docling
pip install docling docling-core
```

#### 2. `CUDA out of memory` Error

**Cause**: GPU doesn't have enough VRAM for the models.

**Solution**:
```python
# Option 1: Use CPU instead
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Option 2: Reduce batch size in embedding_generator.py
# Change batch_size from 32 to 8 or 4
```

#### 3. `torch.cuda.is_available()` returns `False`

**Cause**: PyTorch is installed without CUDA support or CUDA drivers are missing.

**Solution**:
```bash
# Uninstall existing torch
pip uninstall torch torchvision torchaudio

# Reinstall with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Slow PDF Processing

**Cause**: Processing on CPU is inherently slower.

**Solutions**:
- Use a GPU if available
- Process PDFs in batches overnight
- For very large documents, consider splitting them first

#### 5. `OSError: [Errno 28] No space left on device`

**Cause**: Disk is full (models download ~5GB on first run).

**Solution**:
```bash
# Check disk space
df -h  # Linux/macOS
wmic logicaldisk get size,freespace,caption  # Windows

# Clear Hugging Face cache if needed
rm -rf ~/.cache/huggingface/  # Linux/macOS
rmdir /s /q %USERPROFILE%\.cache\huggingface  # Windows
```

#### 6. `ConnectionError` during model download

**Cause**: Network issues or firewall blocking Hugging Face.

**Solution**:
```bash
# Set proxy if behind corporate firewall
set HTTP_PROXY=http://proxy:port  # Windows
export HTTP_PROXY=http://proxy:port  # Linux/macOS

# Or download models manually and set cache directory
set HF_HOME=C:\path\to\models  # Windows
export HF_HOME=/path/to/models  # Linux/macOS
```

#### 7. Poor OCR Results on Scanned PDFs

**Cause**: Low-quality scans or unusual fonts.

**Solutions**:
- Ensure PDFs are at least 300 DPI
- Pre-process images to improve contrast
- For handwritten text, results may be limited

#### 8. `ChromaDB` Collection Already Exists Error

**Cause**: Trying to recreate an existing collection.

**Solution**:
```bash
# Delete existing vector store and re-ingest
rm -rf data/vector_store/*  # Linux/macOS
rmdir /s /q data\vector_store  # Windows
mkdir data\vector_store
```

#### 9. `PermissionError: [Errno 13] Permission denied`

**Cause**: File is open in another program or lacks write permissions.

**Solution**:
- Close any PDF viewers or applications using the files
- Run terminal as Administrator (Windows) or use `sudo` (Linux/macOS)

#### 10. Empty Query Results

**Cause**: Documents not ingested or query doesn't match content.

**Solution**:
```bash
# Check if documents are ingested
python -m src.rag_pipeline --stats

# If count is 0, run ingestion
python -m src.rag_pipeline --ingest

# Try broader search terms
python -m src.rag_pipeline --query "summary"
```

### Getting Help

If you encounter issues not listed here:

1. **Check the logs**: Run with verbose logging
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Open an issue**: [GitHub Issues](https://github.com/Sreeram68/RAG-PDF-docling/issues)

3. **Include this information**:
   - Python version (`python --version`)
   - Operating system
   - GPU info (`nvidia-smi` if applicable)
   - Full error traceback
   - Steps to reproduce

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| docling | ≥2.15.0 | PDF extraction with OCR |
| docling-core | ≥2.21.0 | Core Docling components |
| FlagEmbedding | ≥1.2.0 | BGE-M3 embeddings |
| chromadb | ≥0.4.0 | Vector database |
| torch | ≥2.0.0 | Deep learning framework |
| transformers | ≥4.35.0 | Hugging Face transformers |
| numpy | ≥1.24.0 | Numerical operations |
| tqdm | ≥4.65.0 | Progress bars |

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU device(s) to use | All available |
| `HF_HOME` | Hugging Face cache directory | `~/.cache/huggingface` |
| `TOKENIZERS_PARALLELISM` | Tokenizer parallelism | `true` |

### Pipeline Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `chunk_size` | Maximum chunk size in characters | 1000 |
| `chunk_overlap` | Overlap between chunks | 200 |
| `top_k` | Number of results to retrieve | 5 |
| `enable_ocr` | Enable OCR for scanned docs | True |
| `enable_table_extraction` | Extract tables from PDFs | True |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [IBM Docling](https://github.com/DS4SD/docling) - Document understanding library
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Multilingual embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Hugging Face](https://huggingface.co/) - Model hosting and transformers

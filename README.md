# RAG Pipeline with Docling and BGE-M3

A robust local RAG (Retrieval-Augmented Generation) pipeline that handles scanned PDFs and complex tables without data loss.

## Features

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

## Project Structure

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
├── requirements.txt
└── README.md
```

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. GPU Support (Recommended)

For faster inference with GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### 1. Add Documents

Place your PDF files in the `data/source_pdfs/` directory.

### 2. Ingest Documents

```bash
python -m src.rag_pipeline --ingest
```

This will:
1. Process all PDFs using Docling
2. Convert to Markdown (saved in `data/processed_markdown/`)
3. Chunk text with semantic boundaries
4. Generate BGE-M3 embeddings
5. Store in ChromaDB vector database

### 3. Query the System

```bash
python -m src.rag_pipeline --query "What is the quarterly revenue?"
```

### 4. Check Statistics

```bash
python -m src.rag_pipeline --stats
```

## Python API

```python
from src.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Ingest documents
stats = pipeline.ingest_documents()
print(f"Processed {stats['pdfs_processed']} PDFs")

# Query
result = pipeline.query("What are the key findings?")
print(result.context)

# Get formatted prompt for LLM
prompt = pipeline.get_context_for_llm("Summarize the main points")
# Send 'prompt' to your LLM (Gemma 3 12B, GPT-4, etc.)
```

## Configuration

### RAG Pipeline Options

```python
pipeline = RAGPipeline(
    source_dir="data/source_pdfs",       # PDF source directory
    output_dir="data/processed_markdown", # Markdown output
    vector_store_dir="data/vector_store", # Vector DB location
    chunk_size=1000,                       # Characters per chunk
    chunk_overlap=200,                     # Overlap between chunks
    top_k=5                                # Results to retrieve
)
```

### Document Processor Options

```python
from src.document_processor import DocumentProcessor

processor = DocumentProcessor(
    enable_ocr=True,              # Enable OCR for scanned docs
    enable_table_extraction=True  # Extract table structure
)
```

## Why Docling + BGE-M3?

### Docling Advantages
- **Vision-based layout analysis**: Understands where tables, headers, and images sit on a page
- **Superior to PyPDF**: Handles complex layouts that break traditional parsers
- **Clean Markdown output**: Perfect format for LLMs like Gemma 3 12B
- **Table preservation**: Maintains table structure for accurate retrieval

### BGE-M3 Advantages
- **Multi-granularity**: Dense, sparse, and multi-vector embeddings
- **Long context**: Supports up to 8192 tokens
- **Multilingual**: Works across 100+ languages
- **State-of-the-art**: Top performance on MTEB benchmarks

## Integration with LLMs

### With Ollama (Gemma 3 12B)

```python
import requests

# Get context from RAG
result = pipeline.query("What are the financial results?")

# Send to Ollama
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "gemma3:12b",
        "prompt": pipeline.get_context_for_llm("What are the financial results?"),
        "stream": False
    }
)
print(response.json()["response"])
```

## License

MIT License

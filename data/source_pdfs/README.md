# Source PDF Files

Place your PDF documents in this folder for processing by the RAG pipeline.

## Supported Formats

- Standard PDF files (`.pdf`)
- Scanned PDFs (OCR will be applied automatically)
- PDFs with complex tables and layouts

## Usage

1. Copy your PDF files to this directory
2. Run the ingestion process:
   ```bash
   python -m src.rag_pipeline --ingest
   ```

The pipeline will:
1. Extract text, tables, and structure from each PDF using Docling
2. Convert content to clean Markdown
3. Chunk the content intelligently
4. Generate BGE-M3 embeddings
5. Store in the vector database for retrieval

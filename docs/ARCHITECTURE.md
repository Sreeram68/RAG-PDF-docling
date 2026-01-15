# RAG Pipeline Architecture

This document describes the design, concepts, and logical flow of the RAG (Retrieval-Augmented Generation) pipeline.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RAG-PDF-docling Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │   Docling    │───▶│    Text     │───▶│   BGE-M3     │───▶│   ChromaDB  │  │
│   │  (PDF→MD)    │    │   Chunker    │    │  Embeddings  │    │ Vector Store │  │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                   │                    │         │
│         ▼                    ▼                   ▼                    ▼         │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                         RAG Pipeline Orchestrator                        │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│                            ┌──────────────────┐                                 │
│                            │   Ollama LLM     │                                 │
│                            │  (gemma3:12b)    │                                 │
│                            └──────────────────┘                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Source Files                                        │
│                         data/source_pdfs/*.pdf                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DocumentProcessor                                      │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  • IBM Docling - Vision-based document understanding                       │ │
│  │  • OCR for scanned documents                                               │ │
│  │  • Table structure extraction                                              │ │
│  │  • Layout analysis                                                         │ │
│  │  • Export to clean Markdown                                                │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Input:  PDF files (scanned or native)                                          │
│  Output: Markdown files with preserved structure                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         data/processed_markdown/*.md                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             TextChunker                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  Table-Aware Semantic Chunking Strategy:                                   │ │
│  │                                                                            │ │
│  │  1. Parse structural elements (tables, headers, paragraphs)                │ │
│  │  2. Keep tables intact as atomic units (max 4000 chars)                    │ │
│  │  3. Group headers with their content                                       │ │
│  │  4. Split large sections at paragraph/sentence boundaries                  │ │
│  │  5. Merge small adjacent chunks                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Input:  Markdown text                                                          │
│  Output: List of TextChunk objects with metadata                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EmbeddingGenerator                                     │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  BGE-M3 Model (BAAI/bge-m3):                                               │ │
│  │                                                                            │ │
│  │  • 1024-dimensional dense embeddings                                       │ │
│  │  • Multilingual support                                                    │ │
│  │  • Up to 8192 token context                                                │ │
│  │  • GPU-accelerated (CUDA)                                                  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Input:  Text chunks                                                            │
│  Output: 1024-dim embedding vectors                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             VectorStore                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  ChromaDB:                                                                 │ │
│  │                                                                            │ │
│  │  • Persistent local storage                                                │ │
│  │  • Cosine similarity search                                                │ │
│  │  • Metadata filtering                                                      │ │
│  │  • Collection: "documents"                                                 │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Storage: data/vector_store/                                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Document Ingestion Flow

```
                    ┌─────────────────────────────────────┐
                    │         START: Ingestion            │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │   Scan source_pdfs/ directory       │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      For each PDF file:             │
                    └─────────────────────────────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
              ┌──────────────────┐      ┌──────────────────┐
              │  force_reprocess │      │  Check manifest  │
              │     = True?      │      │  (file changed?) │
              └──────────────────┘      └──────────────────┘
                         │                         │
                         │         ┌───────────────┴───────────────┐
                         │         ▼                               ▼
                         │  ┌──────────────┐              ┌──────────────┐
                         │  │   Changed    │              │  No Change   │
                         │  └──────────────┘              └──────────────┘
                         │         │                               │
                         ▼         ▼                               ▼
              ┌─────────────────────────────┐              ┌──────────────┐
              │   Process PDF with Docling  │              │     Skip     │
              │   • OCR (if scanned)        │              └──────────────┘
              │   • Extract tables          │
              │   • Layout analysis         │
              └─────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      Export to Markdown             │
                    │   (preserve table structure)        │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │   Chunk with Table-Aware Strategy   │
                    │   ┌───────────────────────────────┐ │
                    │   │ • Detect tables → keep intact │ │
                    │   │ • Detect headers → group      │ │
                    │   │ • Split large text sections   │ │
                    │   │ • Merge small chunks          │ │
                    │   └───────────────────────────────┘ │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │    Generate BGE-M3 Embeddings       │
                    │         (1024 dimensions)           │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      Store in ChromaDB              │
                    │   • Embeddings                      │
                    │   • Chunk content                   │
                    │   • Metadata (source, type)         │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     Update processing manifest      │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │          END: Ingestion             │
                    └─────────────────────────────────────┘
```

## Query/Retrieval Flow

```
                    ┌─────────────────────────────────────┐
                    │        User Query                   │
                    │  "What is my HbA1c value?"          │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │    Generate Query Embedding         │
                    │         (BGE-M3)                    │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │   Query Classification              │
                    │   ┌───────────────────────────────┐ │
                    │   │ • Health query detection      │ │
                    │   │ • Multi-year comparison?      │ │
                    │   │ • Extract years mentioned     │ │
                    │   └───────────────────────────────┘ │
                    └─────────────────────────────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
              ┌──────────────────┐      ┌──────────────────┐
              │  Single Year     │      │  Multi-Year      │
              │  Standard Search │      │  Comparison      │
              └──────────────────┘      └──────────────────┘
                         │                         │
                         │                         ▼
                         │              ┌──────────────────┐
                         │              │ Per-Year Search  │
                         │              │ • Search "hemoglobin 2021" │
                         │              │ • Search "hemoglobin 2024" │
                         │              │ • Merge & dedupe │
                         │              └──────────────────┘
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │   Vector Similarity Search          │
                    │   ┌───────────────────────────────┐ │
                    │   │ • Cosine similarity           │ │
                    │   │ • top_k = 15-50 results       │ │
                    │   │ • threshold = 0.05-0.2       │ │
                    │   └───────────────────────────────┘ │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     Retrieved Chunks                │
                    │  ┌─────────────────────────────────┐│
                    │  │ Chunk 1: Health check table    ││
                    │  │ | Test | Result | Unit |       ││
                    │  │ | HbA1c | 5.7 | % |            ││
                    │  │ Score: 0.604                   ││
                    │  ├─────────────────────────────────┤│
                    │  │ Chunk 2: Reference ranges...   ││
                    │  │ Score: 0.582                   ││
                    │  └─────────────────────────────────┘│
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      Build LLM Prompt               │
                    │  ┌─────────────────────────────────┐│
                    │  │ System: You are a medical      ││
                    │  │ records assistant...           ││
                    │  │                                ││
                    │  │ Context: [retrieved chunks]    ││
                    │  │                                ││
                    │  │ Question: What is my HbA1c?    ││
                    │  └─────────────────────────────────┘│
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       Ollama LLM (gemma3:12b)       │
                    │                                     │
                    │   Local inference - no API calls    │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │           Answer                    │
                    │  ┌─────────────────────────────────┐│
                    │  │  "5.7%"                        ││
                    │  │                                ││
                    │  │  Sources: Apollo Healthcheck   ││
                    │  │  Confidence: 0.604             ││
                    │  └─────────────────────────────────┘│
                    └─────────────────────────────────────┘
```

## Table-Aware Chunking Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        BEFORE: Naive Chunking                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Original Table:                                                               │
│   ┌────────────────────────────────────────────────┐                            │
│   │ | Test    | Result | Unit  | Reference |      │                             │
│   │ |---------|--------|-------|-----------|      │                             │
│   │ | HbA1c   | 5.7    | %     | <5.7      |      │                             │
│   │ | Glucose | 106    | mg/dL | 70-110    |      │                             │
│   └────────────────────────────────────────────────┘                            │
│                                                                                 │
│   ❌ Split into chunks (BROKEN):                                                │
│   ┌──────────────────────┐  ┌──────────────────────┐                            │
│   │ Chunk 1:             │  │ Chunk 2:             │                            │
│   │ | Test | Result |    │  │ | Glucose | 106 |   │                             │
│   │ |------|--------|    │  │                      │                            │
│   │ | HbA1c | 5.7 |      │  │ (missing context!)   │                            │
│   └──────────────────────┘  └──────────────────────┘                            │
│                                                                                 │
│   Problem: Row split from header, context lost                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                        AFTER: Table-Aware Chunking                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ✅ Table kept as single chunk:                                                │
│   ┌────────────────────────────────────────────────┐                            │
│   │ Chunk 1 (type: table):                         │                            │
│   │ | Test    | Result | Unit  | Reference |      │                            │
│   │ |---------|--------|-------|-----------|      │                            │
│   │ | HbA1c   | 5.7    | %     | <5.7      |      │                            │
│   │ | Glucose | 106    | mg/dL | 70-110    |      │                            │
│   │                                                │                            │
│   │ Metadata: {"type": "table"}                    │                            │
│   └────────────────────────────────────────────────┘                            │
│                                                                                  │
│   Benefits:                                                                      │
│   • Complete context preserved                                                   │
│   • Headers stay with data rows                                                  │
│   • Better semantic matching                                                     │
│   • Accurate value retrieval                                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Multi-Year Comparison Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│               Query: "Compare my hemoglobin between 2021 and 2024"              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Query Classification                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  1. Detect health keywords: "hemoglobin" ✓                                 │ │
│  │  2. Detect comparison: "compare" or multiple years ✓                       │ │
│  │  3. Extract years: [2021, 2024]                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Per-Year Separate Searches                               │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐            │
│  │ Search: "hemoglobin 2021"   │  │ Search: "hemoglobin 2024"   │            │
│  │ Filter: source contains     │  │ Filter: source contains     │            │
│  │         "2021"              │  │         "2024"              │            │
│  │ Results: 10 chunks from     │  │ Results: 5 chunks from      │            │
│  │ 2021 health check           │  │ 2024 health check           │            │
│  └──────────────────────────────┘  └──────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Interleaved Context Building                            │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  Position 0: 2021 chunk (hemoglobin: 15.3 g/dL)                            │ │
│  │  Position 1: 2024 chunk (hemoglobin: 14.9 g/dL)                            │ │
│  │  Position 2: 2021 chunk (other blood tests)                                │ │
│  │  Position 3: 2024 chunk (other blood tests)                                │ │
│  │  ...                                                                       │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Benefits: LLM sees data from BOTH years early in context                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Comparison-Optimized Prompt                            │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  "Find the requested test values for EACH year mentioned.                  │ │
│  │   Look carefully at EVERY document section.                                │ │
│  │   Report values from all years found."                                     │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               LLM Answer                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  "2021: Hemoglobin was 15.3 g/dL                                           │ │
│  │   2024: Hemoglobin was 14.9 g/dL"                                          │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│                            rag_pipeline.py                                       │
│                           (Main Orchestrator)                                    │
│                                  │                                               │
│              ┌───────────────────┼───────────────────┐                          │
│              │                   │                   │                          │
│              ▼                   ▼                   ▼                          │
│   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐               │
│   │document_processor│ │  text_chunker    │ │    retriever     │               │
│   │      .py         │ │      .py         │ │      .py         │               │
│   └────────┬─────────┘ └──────────────────┘ └────────┬─────────┘               │
│            │                                          │                          │
│            │                                          ▼                          │
│            │                                ┌──────────────────┐                │
│            │                                │  vector_store    │                │
│            │                                │      .py         │                │
│            │                                └────────┬─────────┘                │
│            │                                         │                          │
│            │                   ┌─────────────────────┘                          │
│            │                   │                                                 │
│            ▼                   ▼                                                 │
│   ┌──────────────────────────────────────────┐                                  │
│   │          embedding_generator.py           │                                  │
│   │             (BGE-M3 Model)                │                                  │
│   └──────────────────────────────────────────┘                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  PDF    │───▶│Markdown │───▶│ Chunks  │───▶│Embeddings───▶│ChromaDB │
│ Files   │    │  Files  │    │         │    │(vectors)│    │  Store  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │              │
     │              │              │              │              │
   Docling       OCR +         Table-        BGE-M3         Cosine
   Convert      Tables         Aware        1024-dim       Similarity
                Extract       Chunking                      Search
```

## Key Design Decisions

### 1. Why Docling over other PDF libraries?
- **Vision-based understanding**: Sees documents like humans do
- **Superior table extraction**: Preserves complex table structures
- **Built-in OCR**: Handles scanned documents seamlessly
- **Markdown output**: Clean, LLM-friendly format

### 2. Why BGE-M3 embeddings?
- **Multilingual**: Works with 100+ languages
- **Long context**: Handles up to 8192 tokens
- **High quality**: State-of-the-art semantic understanding
- **Dense vectors**: 1024 dimensions for rich representation

### 3. Why table-aware chunking?
- **Preserve context**: Tables are semantic units
- **Accurate retrieval**: Test values found with their labels
- **Better answers**: LLM sees complete table context

### 4. Why local LLM (Ollama)?
- **Privacy**: Data never leaves your machine
- **No API costs**: Run unlimited queries
- **Offline capable**: Works without internet
- **Customizable**: Choose any compatible model

### 5. Why year-aware context retrieval?
- **Multi-year comparisons**: Compare health metrics across different years
- **Intelligent filtering**: Prioritizes documents matching years in query
- **Interleaved context**: Balances data from different years for LLM
- **Health query optimization**: Auto-detects medical queries for better recall

## Performance Characteristics

| Component | Typical Time | Hardware |
|-----------|-------------|----------|
| PDF Processing (Docling) | 5-30s per page | GPU recommended |
| Embedding Generation | 0.1s per chunk | GPU: 10x faster |
| Vector Search | <100ms | CPU sufficient |
| LLM Answer Generation | 2-10s | GPU recommended |

## File Structure

```
RAG-PDF-docling/
├── src/
│   ├── document_processor.py   # Docling PDF extraction
│   ├── text_chunker.py         # Table-aware chunking
│   ├── embedding_generator.py  # BGE-M3 embeddings
│   ├── vector_store.py         # ChromaDB wrapper
│   ├── retriever.py            # Similarity search
│   └── rag_pipeline.py         # Main orchestrator
├── data/
│   ├── source_pdfs/            # Input PDF files
│   ├── processed_markdown/     # Extracted markdown
│   └── vector_store/           # ChromaDB database
├── docs/
│   └── ARCHITECTURE.md         # This document
└── example_usage.py            # CLI interface
```

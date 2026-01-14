"""
Text Chunking Module
Implements intelligent chunking strategies optimized for RAG pipelines.
Preserves semantic coherence and handles Markdown structure.
"""

import re
import logging
from typing import Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    chunk_id: str
    content: str
    source_file: str
    start_char: int
    end_char: int
    metadata: Optional[dict] = None


class TextChunker:
    """
    Intelligently chunks text for RAG retrieval.
    Preserves document structure, tables, and semantic boundaries.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        respect_sentence_boundaries: bool = True,
        respect_markdown_headers: bool = True
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            respect_sentence_boundaries: Avoid cutting mid-sentence
            respect_markdown_headers: Keep sections under headers together
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.respect_markdown_headers = respect_markdown_headers
        
        # Regex patterns
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.table_pattern = re.compile(r'\|.+\|', re.MULTILINE)
        
        logger.info(f"TextChunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_text(self, text: str, source_file: str = "unknown") -> list[TextChunk]:
        """
        Chunk text into smaller pieces for embedding.
        
        Args:
            text: Full text content to chunk
            source_file: Name of source file for metadata
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        if self.respect_markdown_headers:
            return self._chunk_by_headers(text, source_file)
        else:
            return self._chunk_by_size(text, source_file)
    
    def _chunk_by_headers(self, text: str, source_file: str) -> list[TextChunk]:
        """
        Chunk text respecting Markdown header structure.
        Keeps content under headers together when possible.
        """
        chunks = []
        
        # Split by headers
        sections = self._split_by_headers(text)
        
        chunk_idx = 0
        for section in sections:
            # If section is small enough, keep it as one chunk
            if len(section) <= self.chunk_size:
                chunks.append(TextChunk(
                    chunk_id=f"{source_file}_chunk_{chunk_idx}",
                    content=section.strip(),
                    source_file=source_file,
                    start_char=text.find(section),
                    end_char=text.find(section) + len(section),
                    metadata={"type": "section"}
                ))
                chunk_idx += 1
            else:
                # Section is too large, split further
                sub_chunks = self._chunk_by_size(section, source_file, start_idx=chunk_idx)
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks
    
    def _split_by_headers(self, text: str) -> list[str]:
        """Split text into sections by Markdown headers."""
        # Find all header positions
        headers = list(self.header_pattern.finditer(text))
        
        if not headers:
            return [text]
        
        sections = []
        for i, match in enumerate(headers):
            start = match.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)
        
        # Add any content before first header
        first_header_pos = headers[0].start()
        if first_header_pos > 0:
            preamble = text[:first_header_pos].strip()
            if preamble:
                sections.insert(0, preamble)
        
        return sections
    
    def _chunk_by_size(
        self,
        text: str,
        source_file: str,
        start_idx: int = 0
    ) -> list[TextChunk]:
        """
        Chunk text by size with overlap.
        Respects sentence boundaries when possible.
        """
        chunks = []
        
        if self.respect_sentence_boundaries:
            # Split into sentences first
            sentences = self.sentence_pattern.split(text)
        else:
            sentences = [text]
        
        current_chunk = []
        current_length = 0
        chunk_start = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunks.append(TextChunk(
                    chunk_id=f"{source_file}_chunk_{start_idx + len(chunks)}",
                    content=chunk_content,
                    source_file=source_file,
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_content),
                    metadata={"type": "text"}
                ))
                
                # Apply overlap - keep some sentences for context
                overlap_text = chunk_content[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
                chunk_start = chunk_start + len(chunk_content) - len(overlap_text)
            
            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space
        
        # Add remaining content
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(TextChunk(
                chunk_id=f"{source_file}_chunk_{start_idx + len(chunks)}",
                content=chunk_content,
                source_file=source_file,
                start_char=chunk_start,
                end_char=chunk_start + len(chunk_content),
                metadata={"type": "text"}
            ))
        
        return chunks
    
    def extract_tables(self, text: str) -> list[str]:
        """
        Extract Markdown tables from text.
        Tables are kept intact as separate chunks for better retrieval.
        """
        tables = []
        lines = text.split('\n')
        current_table = []
        in_table = False
        
        for line in lines:
            if self.table_pattern.match(line):
                in_table = True
                current_table.append(line)
            elif in_table and line.strip() == '':
                # End of table
                if current_table:
                    tables.append('\n'.join(current_table))
                    current_table = []
                in_table = False
            elif in_table:
                # Continue table or end it
                if '|' in line or line.strip().startswith('|'):
                    current_table.append(line)
                else:
                    if current_table:
                        tables.append('\n'.join(current_table))
                        current_table = []
                    in_table = False
        
        # Handle table at end of document
        if current_table:
            tables.append('\n'.join(current_table))
        
        return tables


if __name__ == "__main__":
    # Example usage
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    
    sample_text = """
# Introduction

This document describes our quarterly performance metrics.

## Financial Results

The company achieved significant growth in Q4 2024.

| Quarter | Revenue | Growth |
|---------|---------|--------|
| Q1 | $10M | 5% |
| Q2 | $12M | 20% |
| Q3 | $14M | 17% |
| Q4 | $18M | 29% |

## Outlook

We expect continued growth in 2025.
"""
    
    chunks = chunker.chunk_text(sample_text, "sample_doc")
    
    for chunk in chunks:
        print(f"\n{chunk.chunk_id}:\n{chunk.content[:100]}...")

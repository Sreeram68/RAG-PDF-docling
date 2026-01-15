"""
Text Chunking Module
Implements intelligent, table-aware chunking strategies optimized for RAG pipelines.
Preserves semantic coherence, keeps tables intact, and handles Markdown structure.
"""

import re
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of content chunks."""
    TABLE = "table"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    TEXT = "text"


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
    
    Key features:
    - Tables are kept intact as single chunks
    - Headers stay with their content
    - Semantic boundaries respected (paragraphs, sections)
    - Configurable size limits with smart overflow handling
    """
    
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_table_chunk_size: int = 4000,
        respect_tables: bool = True,
        respect_markdown_headers: bool = True
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Overlap between consecutive text chunks
            min_chunk_size: Minimum chunk size (smaller chunks merged)
            max_table_chunk_size: Maximum size for table chunks (large tables split by rows)
            respect_tables: Keep tables as single units when possible
            respect_markdown_headers: Keep sections under headers together
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_table_chunk_size = max_table_chunk_size
        self.respect_tables = respect_tables
        self.respect_markdown_headers = respect_markdown_headers
        
        # Regex patterns
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.table_row_pattern = re.compile(r'^\|.+\|$')
        self.table_separator_pattern = re.compile(r'^\|[-:\s|]+\|$')
        self.paragraph_pattern = re.compile(r'\n\n+')
        
        logger.info(f"TextChunker initialized: size={chunk_size}, overlap={chunk_overlap}, max_table={max_table_chunk_size}")
    
    def chunk_text(self, text: str, source_file: str = "unknown") -> list[TextChunk]:
        """
        Chunk text into semantically coherent pieces for embedding.
        
        Strategy:
        1. Parse document into structural elements (tables, sections, paragraphs)
        2. Keep tables intact as individual chunks
        3. Group headers with their content
        4. Merge small adjacent chunks
        5. Split oversized chunks at paragraph/sentence boundaries
        
        Args:
            text: Full text content to chunk
            source_file: Name of source file for metadata
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Step 1: Parse into structural elements
        elements = self._parse_structural_elements(text)
        
        # Step 2: Create chunks from elements
        chunks = self._create_chunks_from_elements(elements, source_file)
        
        # Step 3: Merge small chunks
        chunks = self._merge_small_chunks(chunks, source_file)
        
        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks
    
    def _parse_structural_elements(self, text: str) -> list[dict]:
        """
        Parse text into structural elements: tables, headers+content, paragraphs.
        
        Returns list of dicts with 'type', 'content', 'start', 'end' keys.
        """
        elements = []
        lines = text.split('\n')
        
        i = 0
        current_pos = 0
        
        while i < len(lines):
            line = lines[i]
            line_start = current_pos
            
            # Check for table
            if self._is_table_row(line):
                table_lines = []
                table_start = current_pos
                
                # Collect all table rows
                while i < len(lines) and (self._is_table_row(lines[i]) or 
                                          self._is_table_separator(lines[i]) or
                                          lines[i].strip() == ''):
                    if lines[i].strip():  # Skip empty lines within table detection
                        table_lines.append(lines[i])
                    current_pos += len(lines[i]) + 1  # +1 for newline
                    i += 1
                    
                    # Stop if we hit two consecutive non-table lines
                    if i < len(lines) and not self._is_table_row(lines[i]) and lines[i].strip():
                        break
                
                if table_lines:
                    table_content = '\n'.join(table_lines)
                    elements.append({
                        'type': ChunkType.TABLE,
                        'content': table_content,
                        'start': table_start,
                        'end': current_pos
                    })
                continue
            
            # Check for header
            header_match = self.header_pattern.match(line)
            if header_match:
                header_level = len(header_match.group(1))
                section_lines = [line]
                current_pos += len(line) + 1
                i += 1
                
                # Collect content until next header of same or higher level
                while i < len(lines):
                    next_line = lines[i]
                    next_header = self.header_pattern.match(next_line)
                    
                    if next_header and len(next_header.group(1)) <= header_level:
                        break  # Stop at same or higher level header
                    
                    # Check if this starts a table - don't include in section
                    if self._is_table_row(next_line):
                        break
                    
                    section_lines.append(next_line)
                    current_pos += len(next_line) + 1
                    i += 1
                
                section_content = '\n'.join(section_lines)
                if section_content.strip():
                    elements.append({
                        'type': ChunkType.SECTION,
                        'content': section_content.strip(),
                        'start': line_start,
                        'end': current_pos,
                        'header_level': header_level
                    })
                continue
            
            # Regular paragraph/text
            para_lines = [line]
            current_pos += len(line) + 1
            i += 1
            
            # Collect until we hit a table, header, or double newline
            while i < len(lines):
                next_line = lines[i]
                
                if self._is_table_row(next_line) or self.header_pattern.match(next_line):
                    break
                
                # Stop on paragraph break (empty line followed by content)
                if not next_line.strip() and i + 1 < len(lines) and lines[i + 1].strip():
                    para_lines.append(next_line)
                    current_pos += len(next_line) + 1
                    i += 1
                    break
                
                para_lines.append(next_line)
                current_pos += len(next_line) + 1
                i += 1
            
            para_content = '\n'.join(para_lines)
            if para_content.strip():
                elements.append({
                    'type': ChunkType.PARAGRAPH,
                    'content': para_content.strip(),
                    'start': line_start,
                    'end': current_pos
                })
        
        return elements
    
    def _is_table_row(self, line: str) -> bool:
        """Check if a line is a markdown table row."""
        stripped = line.strip()
        return bool(stripped and stripped.startswith('|') and stripped.endswith('|'))
    
    def _is_table_separator(self, line: str) -> bool:
        """Check if a line is a markdown table separator (|---|---|)."""
        return bool(self.table_separator_pattern.match(line.strip()))
    
    def _create_chunks_from_elements(
        self, 
        elements: list[dict], 
        source_file: str
    ) -> list[TextChunk]:
        """Convert structural elements into chunks."""
        chunks = []
        chunk_idx = 0
        
        for element in elements:
            content = element['content']
            elem_type = element['type']
            
            # Tables: keep intact or split by rows if too large
            if elem_type == ChunkType.TABLE:
                table_chunks = self._chunk_table(content, source_file, chunk_idx)
                chunks.extend(table_chunks)
                chunk_idx += len(table_chunks)
            
            # Sections/paragraphs: split if too large
            elif len(content) <= self.chunk_size:
                chunks.append(TextChunk(
                    chunk_id=f"{source_file}_chunk_{chunk_idx}",
                    content=content,
                    source_file=source_file,
                    start_char=element['start'],
                    end_char=element['end'],
                    metadata={"type": elem_type.value}
                ))
                chunk_idx += 1
            else:
                # Split large sections at paragraph boundaries
                sub_chunks = self._split_large_text(content, source_file, chunk_idx, elem_type)
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
        
        return chunks
    
    def _chunk_table(
        self, 
        table_content: str, 
        source_file: str, 
        start_idx: int
    ) -> list[TextChunk]:
        """
        Chunk a table, keeping it intact if possible.
        For very large tables, split by rows while preserving header.
        """
        if len(table_content) <= self.max_table_chunk_size:
            return [TextChunk(
                chunk_id=f"{source_file}_chunk_{start_idx}",
                content=table_content,
                source_file=source_file,
                start_char=0,
                end_char=len(table_content),
                metadata={"type": "table"}
            )]
        
        # Large table - split by rows, keeping header with each chunk
        lines = table_content.split('\n')
        chunks = []
        
        # Extract header rows (first row + separator)
        header_lines = []
        data_lines = []
        header_done = False
        
        for line in lines:
            if not header_done:
                header_lines.append(line)
                if self._is_table_separator(line):
                    header_done = True
            else:
                data_lines.append(line)
        
        header = '\n'.join(header_lines)
        header_size = len(header) + 1  # +1 for newline
        
        # Split data rows into chunks
        current_rows = []
        current_size = header_size
        
        for row in data_lines:
            row_size = len(row) + 1
            
            if current_size + row_size > self.max_table_chunk_size and current_rows:
                # Create chunk with header + current rows
                chunk_content = header + '\n' + '\n'.join(current_rows)
                chunks.append(TextChunk(
                    chunk_id=f"{source_file}_chunk_{start_idx + len(chunks)}",
                    content=chunk_content,
                    source_file=source_file,
                    start_char=0,
                    end_char=len(chunk_content),
                    metadata={"type": "table", "split": True}
                ))
                current_rows = []
                current_size = header_size
            
            current_rows.append(row)
            current_size += row_size
        
        # Add remaining rows
        if current_rows:
            chunk_content = header + '\n' + '\n'.join(current_rows)
            chunks.append(TextChunk(
                chunk_id=f"{source_file}_chunk_{start_idx + len(chunks)}",
                content=chunk_content,
                source_file=source_file,
                start_char=0,
                end_char=len(chunk_content),
                metadata={"type": "table", "split": len(chunks) > 0}
            ))
        
        return chunks
    
    def _split_large_text(
        self, 
        text: str, 
        source_file: str, 
        start_idx: int,
        elem_type: ChunkType
    ) -> list[TextChunk]:
        """Split large text at paragraph or sentence boundaries."""
        chunks = []
        
        # Try splitting by paragraphs first
        paragraphs = self.paragraph_pattern.split(text)
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # If single paragraph is too large, split by sentences
            if para_size > self.chunk_size:
                # Save current chunk first
                if current_chunk:
                    chunk_content = '\n\n'.join(current_chunk)
                    chunks.append(TextChunk(
                        chunk_id=f"{source_file}_chunk_{start_idx + len(chunks)}",
                        content=chunk_content,
                        source_file=source_file,
                        start_char=0,
                        end_char=len(chunk_content),
                        metadata={"type": elem_type.value}
                    ))
                    current_chunk = []
                    current_size = 0
                
                # Split paragraph by sentences
                sentence_chunks = self._split_by_sentences(para, source_file, start_idx + len(chunks))
                chunks.extend(sentence_chunks)
                continue
            
            # Check if adding this paragraph exceeds chunk size
            if current_size + para_size + 2 > self.chunk_size and current_chunk:
                chunk_content = '\n\n'.join(current_chunk)
                chunks.append(TextChunk(
                    chunk_id=f"{source_file}_chunk_{start_idx + len(chunks)}",
                    content=chunk_content,
                    source_file=source_file,
                    start_char=0,
                    end_char=len(chunk_content),
                    metadata={"type": elem_type.value}
                ))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1][-self.chunk_overlap:]
                    current_chunk = [overlap_text] if overlap_text else []
                    current_size = len(overlap_text) if overlap_text else 0
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(para)
            current_size += para_size + 2
        
        # Add remaining content
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunks.append(TextChunk(
                chunk_id=f"{source_file}_chunk_{start_idx + len(chunks)}",
                content=chunk_content,
                source_file=source_file,
                start_char=0,
                end_char=len(chunk_content),
                metadata={"type": elem_type.value}
            ))
        
        return chunks
    
    def _split_by_sentences(
        self, 
        text: str, 
        source_file: str, 
        start_idx: int
    ) -> list[TextChunk]:
        """Split text by sentences as last resort."""
        chunks = []
        
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sent_size = len(sentence)
            
            if current_size + sent_size + 1 > self.chunk_size and current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunks.append(TextChunk(
                    chunk_id=f"{source_file}_chunk_{start_idx + len(chunks)}",
                    content=chunk_content,
                    source_file=source_file,
                    start_char=0,
                    end_char=len(chunk_content),
                    metadata={"type": "text"}
                ))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sent_size + 1
        
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(TextChunk(
                chunk_id=f"{source_file}_chunk_{start_idx + len(chunks)}",
                content=chunk_content,
                source_file=source_file,
                start_char=0,
                end_char=len(chunk_content),
                metadata={"type": "text"}
            ))
        
        return chunks
    
    def _merge_small_chunks(
        self, 
        chunks: list[TextChunk], 
        source_file: str
    ) -> list[TextChunk]:
        """Merge adjacent small chunks that are below minimum size."""
        if not chunks:
            return chunks
        
        merged = []
        current = None
        
        for chunk in chunks:
            if current is None:
                current = chunk
                continue
            
            # Don't merge tables with non-tables
            current_is_table = current.metadata and current.metadata.get("type") == "table"
            chunk_is_table = chunk.metadata and chunk.metadata.get("type") == "table"
            
            if current_is_table != chunk_is_table:
                merged.append(current)
                current = chunk
                continue
            
            # Merge if combined size is reasonable and current is small
            combined_size = len(current.content) + len(chunk.content) + 2
            
            if len(current.content) < self.min_chunk_size and combined_size <= self.chunk_size:
                # Merge chunks
                current = TextChunk(
                    chunk_id=current.chunk_id,
                    content=current.content + '\n\n' + chunk.content,
                    source_file=source_file,
                    start_char=current.start_char,
                    end_char=chunk.end_char,
                    metadata=current.metadata
                )
            else:
                merged.append(current)
                current = chunk
        
        if current:
            merged.append(current)
        
        # Re-index chunk IDs
        for i, chunk in enumerate(merged):
            chunk.chunk_id = f"{source_file}_chunk_{i}"
        
        return merged


if __name__ == "__main__":
    # Example usage
    chunker = TextChunker(chunk_size=1500, chunk_overlap=200)
    
    sample_text = """
# Introduction

This document describes our quarterly performance metrics. We have seen significant improvements across all key indicators.

## Financial Results

The company achieved significant growth in Q4 2024.

| Quarter | Revenue | Growth | Profit |
|---------|---------|--------|--------|
| Q1 | $10M | 5% | $1M |
| Q2 | $12M | 20% | $1.5M |
| Q3 | $14M | 17% | $2M |
| Q4 | $18M | 29% | $3M |

## Health Check Results

| Test Name | Result | Unit | Reference Range |
|-----------|--------|------|-----------------|
| GLUCOSE, FASTING | 106 | mg/dL | 70-110 |
| HBA1C | 5.7 | % | <5.7 |
| CHOLESTEROL | 180 | mg/dL | <200 |

## Outlook

We expect continued growth in 2025. The market conditions remain favorable.
"""
    
    chunks = chunker.chunk_text(sample_text, "sample_doc")
    
    print(f"\nTotal chunks: {len(chunks)}\n")
    for chunk in chunks:
        chunk_type = chunk.metadata.get("type", "unknown") if chunk.metadata else "unknown"
        print(f"{'='*60}")
        print(f"Chunk: {chunk.chunk_id} | Type: {chunk_type} | Size: {len(chunk.content)}")
        print(f"{'='*60}")
        print(chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content)
        print()

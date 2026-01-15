"""
Document Processor Module
Uses Docling to extract text, tables, and structured content from PDFs.
Converts documents to clean Markdown format for optimal LLM consumption.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
from datetime import datetime

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Represents a processed document with extracted content."""
    source_path: str
    markdown_content: str
    title: str
    num_pages: int
    tables_count: int
    images_count: int


class DocumentProcessor:
    """
    Handles PDF document processing using Docling.
    Extracts text, tables, and images while preserving document structure.
    """
    
    def __init__(
        self,
        source_dir: str = "data/source_pdfs",
        output_dir: str = "data/processed_markdown",
        enable_ocr: bool = True,
        enable_table_extraction: bool = True
    ):
        """
        Initialize the document processor.
        
        Args:
            source_dir: Directory containing source PDF files
            output_dir: Directory to save processed Markdown files
            enable_ocr: Enable OCR for scanned documents
            enable_table_extraction: Enable table structure extraction
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.enable_ocr = enable_ocr
        self.enable_table_extraction = enable_table_extraction
        
        # Create directories if they don't exist
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File tracking manifest
        self.manifest_path = self.output_dir / ".processed_manifest.json"
        self.manifest = self._load_manifest()
        
        # Configure Docling pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = enable_ocr
        pipeline_options.do_table_structure = enable_table_extraction
        
        # Initialize the document converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        logger.info(f"DocumentProcessor initialized. Source: {self.source_dir}, Output: {self.output_dir}")
    
    def _load_manifest(self) -> dict:
        """Load the processing manifest from disk."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Could not load manifest, starting fresh")
        return {"processed_files": {}}
    
    def _save_manifest(self):
        """Save the processing manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file for change detection."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_file_info(self, file_path: Path) -> dict:
        """Get file metadata for tracking."""
        stat = file_path.stat()
        return {
            "path": str(file_path.absolute()),
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
            "hash": self._get_file_hash(file_path),
            "processed_at": datetime.now().isoformat()
        }
    
    def _needs_processing(self, pdf_path: Path) -> bool:
        """
        Check if a PDF needs to be processed.
        Returns True if file is new or has been modified.
        """
        file_key = str(pdf_path.absolute())
        
        if file_key not in self.manifest["processed_files"]:
            logger.info(f"New file detected: {pdf_path.name}")
            return True
        
        stored_info = self.manifest["processed_files"][file_key]
        current_stat = pdf_path.stat()
        
        # Quick check: modification time and size
        if (current_stat.st_mtime != stored_info.get("modified_time") or
            current_stat.st_size != stored_info.get("size")):
            # Verify with hash to confirm actual content change
            current_hash = self._get_file_hash(pdf_path)
            if current_hash != stored_info.get("hash"):
                logger.info(f"Modified file detected: {pdf_path.name}")
                return True
        
        logger.debug(f"Skipping already processed: {pdf_path.name}")
        return False
    
    def _mark_as_processed(self, pdf_path: Path):
        """Mark a file as processed in the manifest."""
        file_key = str(pdf_path.absolute())
        self.manifest["processed_files"][file_key] = self._get_file_info(pdf_path)
        self._save_manifest()
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> ProcessedDocument:
        """
        Process a single PDF file and convert to Markdown.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessedDocument with extracted content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Processing: {pdf_path.name}")
        
        # Convert PDF using Docling
        result = self.converter.convert(str(pdf_path))
        
        # Export to Markdown format
        markdown_content = result.document.export_to_markdown()
        
        # Count extracted elements - handle both properties and methods
        tables_count = len(result.document.tables) if hasattr(result.document, 'tables') else 0
        images_count = len(result.document.pictures) if hasattr(result.document, 'pictures') else 0
        
        # Handle num_pages which might be a method or property
        if hasattr(result.document, 'num_pages'):
            np = result.document.num_pages
            num_pages = np() if callable(np) else np
        else:
            num_pages = len(result.document.pages) if hasattr(result.document, 'pages') else 0
        
        # Create processed document
        processed_doc = ProcessedDocument(
            source_path=str(pdf_path),
            markdown_content=markdown_content,
            title=pdf_path.stem,
            num_pages=num_pages,
            tables_count=tables_count,
            images_count=images_count
        )
        
        logger.info(f"Processed {pdf_path.name}: {num_pages} pages, {tables_count} tables, {images_count} images")
        
        return processed_doc
    
    def save_markdown(self, doc: ProcessedDocument) -> str:
        """
        Save processed document as Markdown file.
        
        Args:
            doc: ProcessedDocument to save
            
        Returns:
            Path to saved Markdown file
        """
        output_path = self.output_dir / f"{doc.title}.md"
        
        # Add metadata header
        header = f"""---
source: {doc.source_path}
title: {doc.title}
pages: {doc.num_pages}
tables: {doc.tables_count}
images: {doc.images_count}
---

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header + doc.markdown_content)
        
        logger.info(f"Saved: {output_path}")
        return str(output_path)
    
    def process_directory(self, force_reprocess: bool = False) -> list[ProcessedDocument]:
        """
        Process all PDFs in the source directory.
        Only processes new or modified files unless force_reprocess=True.
        
        Args:
            force_reprocess: If True, reprocess all files regardless of status
        
        Returns:
            List of ProcessedDocument objects
        """
        pdf_files = list(self.source_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.source_dir}")
            return []
        
        # Filter to only files that need processing
        if force_reprocess:
            files_to_process = pdf_files
            logger.info(f"Force reprocessing all {len(pdf_files)} PDF files")
        else:
            files_to_process = [f for f in pdf_files if self._needs_processing(f)]
            skipped = len(pdf_files) - len(files_to_process)
            if skipped > 0:
                logger.info(f"Skipping {skipped} already processed files")
        
        if not files_to_process:
            logger.info("No new or modified PDFs to process")
            return []
        
        logger.info(f"Processing {len(files_to_process)} PDF files")
        
        processed_docs = []
        for pdf_path in files_to_process:
            try:
                doc = self.process_pdf(pdf_path)
                self.save_markdown(doc)
                self._mark_as_processed(pdf_path)  # Track successful processing
                processed_docs.append(doc)
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_docs)}/{len(files_to_process)} documents")
        return processed_docs
    
    def get_processing_status(self) -> dict:
        """
        Get status of all PDF files.
        
        Returns:
            Dict with processed and pending file counts
        """
        pdf_files = list(self.source_dir.glob("*.pdf"))
        pending = [f for f in pdf_files if self._needs_processing(f)]
        
        return {
            "total_pdfs": len(pdf_files),
            "already_processed": len(pdf_files) - len(pending),
            "pending": len(pending),
            "pending_files": [f.name for f in pending]
        }


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    
    # Process all PDFs in the source directory
    documents = processor.process_directory()
    
    for doc in documents:
        print(f"Processed: {doc.title} ({doc.num_pages} pages)")

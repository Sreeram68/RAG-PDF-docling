"""
RAG Pipeline Module
Main orchestrator that combines document processing, embedding, and retrieval.
Provides a complete pipeline for building a RAG-powered Q&A system.
"""

import logging
import requests
from pathlib import Path
from typing import Optional

from .document_processor import DocumentProcessor
from .text_chunker import TextChunker
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .retriever import Retriever, RetrievalResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline orchestrating:
    1. PDF extraction with Docling
    2. Text chunking with semantic boundaries
    3. Embedding generation with BGE-M3
    4. Vector storage with ChromaDB
    5. Retrieval for question answering
    """
    
    def __init__(
        self,
        source_dir: str = "data/source_pdfs",
        output_dir: str = "data/processed_markdown",
        vector_store_dir: str = "data/vector_store",
        chunk_size: int = 1500,  # Increased to capture wider tables from health reports
        chunk_overlap: int = 300,
        top_k: int = 5
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            source_dir: Directory containing source PDF files
            output_dir: Directory for processed Markdown files
            vector_store_dir: Directory for vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
        """
        logger.info("Initializing RAG Pipeline...")
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            source_dir=source_dir,
            output_dir=output_dir
        )
        
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.embedding_generator = EmbeddingGenerator()
        
        self.vector_store = VectorStore(
            persist_directory=vector_store_dir
        )
        
        self.retriever = Retriever(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
            top_k=top_k
        )
        
        logger.info("RAG Pipeline initialized successfully")
    
    def ingest_documents(self, force_reprocess: bool = False) -> dict:
        """
        Process all PDFs and add to vector store.
        Only processes new or modified files unless force_reprocess=True.
        
        Args:
            force_reprocess: If True, reprocess all files regardless of status
        
        Returns:
            Statistics about ingested documents
        """
        logger.info("Starting document ingestion...")
        
        stats = {
            "pdfs_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": []
        }
        
        # Process new/modified PDFs in source directory
        processed_docs = self.document_processor.process_directory(force_reprocess=force_reprocess)
        stats["pdfs_processed"] = len(processed_docs)
        
        for doc in processed_docs:
            try:
                # Chunk the document
                chunks = self.text_chunker.chunk_text(
                    doc.markdown_content,
                    source_file=doc.title
                )
                stats["chunks_created"] += len(chunks)
                
                if not chunks:
                    continue
                
                # Generate embeddings
                texts = [chunk.content for chunk in chunks]
                embeddings = self.embedding_generator.generate_embeddings(texts)
                stats["embeddings_generated"] += len(embeddings)
                
                # Add to vector store
                self.vector_store.add_documents(
                    chunk_ids=[chunk.chunk_id for chunk in chunks],
                    texts=texts,
                    embeddings=[emb.dense_embedding for emb in embeddings],
                    source_files=[chunk.source_file for chunk in chunks],
                    metadatas=[chunk.metadata or {} for chunk in chunks]
                )
                
                logger.info(f"Ingested {doc.title}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error ingesting {doc.title}: {e}")
                stats["errors"].append({"file": doc.title, "error": str(e)})
        
        logger.info(f"Ingestion complete: {stats['pdfs_processed']} PDFs, {stats['chunks_created']} chunks")
        return stats
    
    def ingest_single_pdf(self, pdf_path: str) -> dict:
        """
        Process a single PDF and add to vector store.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Statistics about ingested document
        """
        logger.info(f"Ingesting single PDF: {pdf_path}")
        
        # Process PDF
        doc = self.document_processor.process_pdf(pdf_path)
        self.document_processor.save_markdown(doc)
        
        # Chunk the document
        chunks = self.text_chunker.chunk_text(
            doc.markdown_content,
            source_file=doc.title
        )
        
        if not chunks:
            return {"chunks": 0, "status": "no_content"}
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Add to vector store
        self.vector_store.add_documents(
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            texts=texts,
            embeddings=[emb.dense_embedding for emb in embeddings],
            source_files=[chunk.source_file for chunk in chunks],
            metadatas=[chunk.metadata or {} for chunk in chunks]
        )
        
        return {
            "title": doc.title,
            "pages": doc.num_pages,
            "tables": doc.tables_count,
            "chunks": len(chunks),
            "status": "success"
        }
    
    def query(self, question: str, top_k: Optional[int] = None) -> RetrievalResult:
        """
        Query the RAG system with a question.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            RetrievalResult with relevant context
        """
        return self.retriever.retrieve(question, top_k=top_k)
    
    def get_context_for_llm(self, question: str) -> str:
        """
        Get formatted context for an LLM prompt.
        
        Args:
            question: User question
            
        Returns:
            Formatted context string ready for LLM consumption
        """
        result = self.query(question)
        
        prompt_template = f"""Answer the question based on the following context. 
If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{result.context}

Question: {question}

Answer:"""
        
        return prompt_template
    
    def answer(
        self,
        question: str,
        model: str = "gemma3:12b",
        ollama_url: str = "http://localhost:11434",
        top_k: Optional[int] = None
    ) -> dict:
        """
        Answer a question using retrieved context and an LLM.
        
        Args:
            question: User question
            model: Ollama model to use (default: gemma3:12b)
            ollama_url: Ollama server URL
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with answer, sources, and metadata
        """
        # Detect if this is a health/medical query (for retrieval optimization)
        health_keywords = ['hemoglobin', 'haemoglobin', 'hba1c', 'glucose', 'cholesterol',
                          'blood', 'test', 'medical', 'sugar', 'pressure',
                          'creatinine', 'triglyceride', 'platelet', 'rbc', 'wbc',
                          'iron', 'vitamin', 'thyroid', 'liver', 'kidney']
        # Removed 'health' from keywords to avoid false positives with "healthcheck" filenames
        is_health_query = any(kw in question.lower() for kw in health_keywords)
        
        # Detect financial/business queries
        financial_keywords = ['asset', 'revenue', 'income', 'profit', 'loss', 'equity',
                             'balance sheet', 'cash flow', 'earnings', 'dividend',
                             'liabilities', 'shareholders', 'ebitda', 'margin',
                             'annual report', 'financial', 'fiscal', 'quarter']
        is_financial_query = any(kw in question.lower() for kw in financial_keywords)
        
        # Check for multi-year comparison queries
        years_in_query = [year for year in ['2021', '2022', '2023', '2024', '2025', '2026'] if year in question]
        is_comparison = len(years_in_query) > 1 or 'compare' in question.lower()
        
        # Use more documents for specialized queries to ensure we get all relevant docs
        if top_k:
            k = top_k
        elif is_health_query and is_comparison:
            k = 50  # Need more for multi-year health comparisons
        elif is_financial_query:
            k = 100  # Financial docs have table data that scores low semantically, need more candidates
        elif is_health_query:
            k = 30
        else:
            k = 15
        
        # Temporarily lower threshold to get more context for LLM
        original_threshold = self.retriever.score_threshold
        if is_comparison:
            self.retriever.score_threshold = 0.05
        elif is_health_query or is_financial_query:
            self.retriever.score_threshold = 0.1
        else:
            self.retriever.score_threshold = 0.2
        
        # For multi-year comparison queries, do separate searches per year
        if is_health_query and len(years_in_query) > 1:
            # Extract the core health term from the question
            core_terms = []
            for kw in health_keywords:
                if kw in question.lower():
                    core_terms.append(kw)
            core_query = ' '.join(core_terms) if core_terms else question
            
            # Search for each year separately
            all_results = []
            base_result = None
            for year in years_in_query:
                year_query = f"{core_query} {year}"
                year_result = self.retriever.retrieve(year_query, top_k=20)
                if base_result is None:
                    base_result = year_result
                # Tag results and add
                for r in year_result.results:
                    if year in r.source_file:  # Only add if from matching year
                        all_results.append(r)
            
            # Deduplicate by chunk ID (approximate by content hash)
            seen = set()
            result_results = []
            for r in all_results:
                key = hash(r.content[:200])
                if key not in seen:
                    seen.add(key)
                    result_results.append(r)
            
            # Create a result-like object with combined results
            from src.retriever import RetrievalResult
            context = "\n\n---\n\n".join([f"[From: {r.source_file}]\n{r.content}" for r in result_results[:10]])
            result = RetrievalResult(
                query=question,
                results=result_results,
                context=context
            )
        else:
            # Standard single search
            result = self.retriever.retrieve(question, top_k=k)
        
        # Restore original threshold
        self.retriever.score_threshold = original_threshold
        
        if not result.results:
            return {
                "question": question,
                "answer": "No relevant documents found to answer this question.",
                "sources": [],
                "context_used": ""
            }
        
        # Filter results to prioritize documents matching years mentioned in question
        filtered_results = result.results
        years_mentioned = years_in_query  # Reuse already-extracted years
        
        if is_health_query:
            # For health queries, prioritize health check documents
            health_docs = [r for r in result.results if 'healthcheck' in r.source_file.lower() or 'health' in r.source_file.lower()]
            other_docs = [r for r in result.results if 'healthcheck' not in r.source_file.lower() and 'health' not in r.source_file.lower()]
            
            if years_mentioned:
                # Further filter health docs by year
                year_matched_health = [r for r in health_docs if any(year in r.source_file for year in years_mentioned)]
                other_health = [r for r in health_docs if not any(year in r.source_file for year in years_mentioned)]
                
                # For comparison queries, interleave results from different years
                if is_comparison and len(years_mentioned) > 1:
                    # Group by year
                    by_year = {year: [] for year in years_mentioned}
                    for r in year_matched_health:
                        for year in years_mentioned:
                            if year in r.source_file:
                                by_year[year].append(r)
                                break
                    
                    # Interleave: take from each year alternately
                    interleaved = []
                    max_len = max(len(v) for v in by_year.values()) if by_year else 0
                    for i in range(max_len):
                        for year in years_mentioned:
                            if i < len(by_year[year]):
                                interleaved.append(by_year[year][i])
                    filtered_results = interleaved + other_health + other_docs[:2]
                else:
                    filtered_results = year_matched_health + other_health + other_docs[:2]
            else:
                filtered_results = health_docs + other_docs[:2]
        elif years_mentioned:
            # For non-health queries with years, use existing logic
            matching = [r for r in result.results if any(year in r.source_file for year in years_mentioned)]
            non_matching = [r for r in result.results if not any(year in r.source_file for year in years_mentioned)]
            filtered_results = matching + non_matching[:3]
        
        # Rebuild context with prioritized results
        # Use more chunks for comparison queries to capture data from multiple years
        num_chunks = 15 if is_comparison else 10
        context_parts = []
        for r in filtered_results[:num_chunks]:
            context_parts.append(f"[From: {r.source_file}]\n{r.content}")
        context = "\n\n---\n\n".join(context_parts)
        
        # Truncate if still too long
        if len(context) > 16000:  # Increased for comparison queries
            context = context[:16000] + "\n\n[Context truncated...]"
        
        # Build focused prompt - different for comparison vs single queries
        if is_comparison and is_health_query:
            prompt = f"""You are analyzing medical test results from multiple years. 
Find the requested test values for EACH year mentioned in the question.
Look carefully at EVERY document section - each [From: ...] section may contain data from a different year.
The document name contains the year (e.g., "2021" or "2024").

Documents:
{context}

Question: {question}

Instructions: Find the specific test value for EACH year. Report values from all years found.
Answer:"""
        else:
            prompt = f"""Answer the question based on the medical test results below. Look for test values in tables.
Pay attention to the document source names which contain dates (e.g., "2021" or "2024").

Context:
{context}

Question: {question}
Answer (be brief and direct):"""

        try:
            # Call Ollama API
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for factual answers
                        "num_predict": 300 if is_comparison else 200   # More for comparisons
                    }
                },
                timeout=180  # Increased timeout
            )
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
            
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama. Is it running?")
            return {
                "question": question,
                "answer": "Error: Could not connect to Ollama. Please ensure Ollama is running (run 'ollama serve' in terminal).",
                "sources": [],
                "context_used": result.context
            }
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return {
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "context_used": result.context
            }
        
        # Extract sources
        sources = list(set(r.source_file for r in result.results))
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_chunks_used": len(result.results),
            "top_score": result.results[0].score if result.results else 0
        }
    
    def get_stats(self) -> dict:
        """Get current pipeline statistics."""
        processing_status = self.document_processor.get_processing_status()
        return {
            "total_documents": self.vector_store.count(),
            "sources": self.vector_store.get_all_sources(),
            "file_status": processing_status
        }


def main():
    """Main entry point for the RAG pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline with Docling and BGE-M3")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents from source directory")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocess all PDFs (use with --ingest)")
    parser.add_argument("--query", type=str, help="Query the RAG system (returns raw chunks)")
    parser.add_argument("--ask", type=str, help="Ask a question and get a focused answer using LLM")
    parser.add_argument("--model", type=str, default="gemma3:12b", help="Ollama model to use (default: gemma3:12b)")
    parser.add_argument("--stats", action="store_true", help="Show pipeline statistics")
    parser.add_argument("--status", action="store_true", help="Show file processing status")
    
    args = parser.parse_args()
    
    pipeline = RAGPipeline()
    
    if args.status:
        status = pipeline.document_processor.get_processing_status()
        print(f"\n📁 File Processing Status:")
        print(f"   Total PDFs: {status['total_pdfs']}")
        print(f"   Already processed: {status['already_processed']}")
        print(f"   Pending: {status['pending']}")
        if status['pending_files']:
            print(f"   Pending files: {', '.join(status['pending_files'])}")
    
    if args.ingest:
        stats = pipeline.ingest_documents(force_reprocess=args.force_reprocess)
        print(f"Ingestion complete: {stats}")
    
    if args.query:
        result = pipeline.query(args.query)
        print(f"\nQuery: {args.query}")
        print(f"\nRelevant Context:\n{result.context}")
    
    if args.ask:
        print(f"\n🔍 Question: {args.ask}")
        print(f"Using model: {args.model}")
        print("Searching and generating answer...\n")
        
        result = pipeline.answer(args.ask, model=args.model)
        
        print("=" * 60)
        print("📝 ANSWER:")
        print("=" * 60)
        print(result["answer"])
        print("\n" + "-" * 60)
        print(f"📚 Sources: {', '.join(result['sources'])}")
        print(f"📊 Chunks used: {result.get('num_chunks_used', 'N/A')}")
        if isinstance(result.get('top_score'), (int, float)):
            print(f"🎯 Top relevance score: {result.get('top_score'):.3f}")
    
    if args.stats:
        stats = pipeline.get_stats()
        print(f"\n📊 Pipeline Stats:")
        print(f"   Total chunks in vector store: {stats['total_documents']}")
        print(f"   Sources: {', '.join(stats['sources'])}")
        if 'file_status' in stats:
            print(f"   PDFs tracked: {stats['file_status']['total_pdfs']}")


if __name__ == "__main__":
    main()

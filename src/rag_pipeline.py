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
                    metadatas=[chunk.metadata for chunk in chunks]
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
            metadatas=[chunk.metadata for chunk in chunks]
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
        # Use fewer documents to fit in LLM context window
        k = top_k or 5  # Reduced to prevent timeout
        
        # Temporarily lower threshold to get more context for LLM
        original_threshold = self.retriever.score_threshold
        self.retriever.score_threshold = 0.3  # Lower threshold for answer generation
        
        # Retrieve relevant context
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
        
        # Truncate context to prevent LLM timeout (max ~4000 chars)
        context = result.context
        if len(context) > 4000:
            context = context[:4000] + "\n\n[Context truncated...]"
        
        # Build focused prompt for pointed answers
        prompt = f"""Answer the question based on the medical test results below. Look for test values in tables.

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
                        "num_predict": 200   # Limit response length
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

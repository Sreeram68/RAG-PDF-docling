"""
Retriever Module
Handles retrieval of relevant documents for RAG queries.
Supports hybrid search combining dense and sparse retrieval.
"""

import logging
from typing import Optional
from dataclasses import dataclass

from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore, SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieval result with context for RAG."""
    query: str
    results: list[SearchResult]
    context: str  # Formatted context for LLM


class Retriever:
    """
    Retrieves relevant documents for RAG queries.
    Combines embedding generation with vector store search.
    """
    
    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
        top_k: int = 5,
        score_threshold: float = 0.5
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_store: VectorStore instance
            top_k: Number of results to retrieve
            score_threshold: Minimum similarity score threshold
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        logger.info(f"Retriever initialized: top_k={top_k}, threshold={score_threshold}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            top_k: Override default top_k
            filter_source: Optional filter by source file
            
        Returns:
            RetrievalResult with retrieved documents and formatted context
        """
        k = top_k or self.top_k
        
        logger.info(f"Retrieving for query: '{query[:50]}...'")
        
        # Generate query embedding
        query_emb = self.embedding_generator.generate_query_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_emb['dense'],
            top_k=k,
            filter_source=filter_source
        )
        
        # Filter by score threshold
        filtered_results = [r for r in results if r.score >= self.score_threshold]
        
        # Format context for LLM
        context = self._format_context(filtered_results)
        
        logger.info(f"Retrieved {len(filtered_results)} relevant documents")
        
        return RetrievalResult(
            query=query,
            results=filtered_results,
            context=context
        )
    
    def _format_context(self, results: list[SearchResult]) -> str:
        """
        Format retrieved documents as context for LLM.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"""
[Document {i}]
Source: {result.source_file}
Relevance Score: {result.score:.3f}

{result.content}
""")
        
        return "\n---\n".join(context_parts)
    
    def retrieve_with_reranking(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 5
    ) -> RetrievalResult:
        """
        Retrieve documents with two-stage retrieval and reranking.
        First retrieves more candidates, then reranks to get final results.
        
        Args:
            query: User query string
            initial_k: Number of initial candidates to retrieve
            final_k: Final number of results after reranking
            
        Returns:
            RetrievalResult with reranked documents
        """
        # First stage: retrieve more candidates
        initial_result = self.retrieve(query, top_k=initial_k)
        
        if len(initial_result.results) <= final_k:
            return initial_result
        
        # For now, just take top results by score
        # TODO: Implement cross-encoder reranking for better accuracy
        top_results = sorted(
            initial_result.results,
            key=lambda x: x.score,
            reverse=True
        )[:final_k]
        
        context = self._format_context(top_results)
        
        return RetrievalResult(
            query=query,
            results=top_results,
            context=context
        )


if __name__ == "__main__":
    # Example usage
    retriever = Retriever()
    
    # Example query
    result = retriever.retrieve("What is the quarterly revenue growth?")
    
    print(f"Query: {result.query}")
    print(f"Found {len(result.results)} relevant documents")
    print("\nContext for LLM:")
    print(result.context)

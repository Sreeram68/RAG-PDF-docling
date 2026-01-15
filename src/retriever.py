"""
Retriever Module
Handles retrieval of relevant documents for RAG queries.
Supports hybrid search combining dense and sparse retrieval.
"""

import logging
import re
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
        score_threshold: float = 0.5,
        keyword_boost: float = 0.25
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_store: VectorStore instance
            top_k: Number of results to retrieve
            score_threshold: Minimum similarity score threshold
            keyword_boost: Score boost for results containing query keywords
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.keyword_boost = keyword_boost
        
        logger.info(f"Retriever initialized: top_k={top_k}, threshold={score_threshold}")
    
    def _extract_keywords(self, query: str) -> list[str]:
        """
        Extract important keywords from query for hybrid matching.
        
        Args:
            query: User query string
            
        Returns:
            List of keywords to match
        """
        # Remove common stop words and extract meaningful terms
        stop_words = {'what', 'is', 'the', 'a', 'an', 'of', 'in', 'for', 'to', 'and', 
                      'or', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                      'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                      'may', 'might', 'can', 'how', 'when', 'where', 'why', 'which',
                      'this', 'that', 'these', 'those', 'it', 'its', 'with', 'from',
                      'value', 'amount', 'number', 'show', 'tell', 'give', 'find',
                      'much', 'many', 'any', 'some'}
        
        # Extract words (including numbers and hyphenated terms)
        words = re.findall(r'[\w\-]+', query.lower())
        
        # Filter out stop words and short words, keep numbers
        keywords = [w for w in words if (w not in stop_words and len(w) > 2) or w.isdigit()]
        
        return keywords
    
    def _apply_keyword_boost(self, results: list[SearchResult], keywords: list[str]) -> list[SearchResult]:
        """
        Apply score boost to results that contain query keywords.
        This helps tabular data rank higher when it contains exact matches.
        
        Args:
            results: List of search results
            keywords: List of keywords to match
            
        Returns:
            Re-ranked list of search results with boosted scores
        """
        if not keywords:
            return results
        
        # Separate keywords into different categories for smarter matching
        content_keywords = []  # Keywords that should match content
        source_keywords = []   # Keywords that should match source filename
        year_keywords = []     # Year keywords
        
        for kw in keywords:
            if kw.isdigit() and len(kw) == 4:  # Year
                year_keywords.append(kw)
            elif kw.lower() in ['danaher', 'apollo', 'healthcheck']:  # Known document names
                source_keywords.append(kw.lower())
            else:
                content_keywords.append(kw)
        
        boosted_results = []
        for result in results:
            content_lower = result.content.lower()
            source_lower = result.source_file.lower()
            
            # Count content keyword matches (with exact word matching)
            import re
            content_matches = 0
            exact_matches = 0
            for kw in content_keywords:
                if kw in content_lower:
                    content_matches += 1
                    if re.search(rf'\b{re.escape(kw)}\b', content_lower):
                        exact_matches += 1
            
            # Count source matches (document name in filename)
            source_matches = sum(1 for kw in source_keywords if kw in source_lower)
            
            # Count year matches (in content OR filename)
            year_matches = sum(1 for kw in year_keywords if kw in content_lower or kw in source_lower)
            
            # Calculate combined match score
            total_keywords = len(content_keywords) + len(source_keywords) + len(year_keywords)
            if total_keywords == 0:
                boosted_results.append(result)
                continue
            
            total_matches = content_matches + source_matches + year_matches
            
            if total_matches > 0:
                # Weight content matches higher than metadata matches
                content_weight = 2.0
                source_weight = 1.0
                year_weight = 0.5
                
                weighted_score = (content_matches * content_weight + 
                                  source_matches * source_weight + 
                                  year_matches * year_weight)
                max_weighted = (len(content_keywords) * content_weight + 
                               len(source_keywords) * source_weight + 
                               len(year_keywords) * year_weight)
                
                match_ratio = weighted_score / max_weighted if max_weighted > 0 else 0
                
                # Base boost from keyword matches
                boost = self.keyword_boost * match_ratio
                
                # Additional boost for exact word matches in content
                if len(content_keywords) > 0:
                    exact_ratio = exact_matches / len(content_keywords)
                    boost += self.keyword_boost * 0.5 * exact_ratio
                
                # Extra boost if content looks like a table (contains |)
                if '|' in result.content:
                    boost *= 1.5  # 50% extra boost for table content
                
                # Create a new SearchResult with boosted score
                boosted_score = min(1.0, result.score + boost)
                boosted_result = SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=boosted_score,
                    source_file=result.source_file,
                    metadata=result.metadata
                )
                boosted_results.append(boosted_result)
            else:
                boosted_results.append(result)
        
        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        
        return boosted_results
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None,
        use_keyword_boost: bool = True
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query with hybrid search.
        
        Args:
            query: User query string
            top_k: Override default top_k
            filter_source: Optional filter by source file
            use_keyword_boost: Whether to apply keyword-based score boosting
            
        Returns:
            RetrievalResult with retrieved documents and formatted context
        """
        k = top_k or self.top_k
        
        logger.info(f"Retrieving for query: '{query[:50]}...'")
        
        # Extract keywords for hybrid matching
        keywords = self._extract_keywords(query) if use_keyword_boost else []
        if keywords:
            logger.info(f"Keywords for boost: {keywords[:10]}")
        
        # Generate query embedding
        query_emb = self.embedding_generator.generate_query_embedding(query)
        
        # Search vector store - get more results for reranking
        search_k = k * 2 if use_keyword_boost else k
        results = self.vector_store.search(
            query_embedding=query_emb['dense'],
            top_k=search_k,
            filter_source=filter_source
        )
        
        # Apply keyword boost for hybrid search
        if use_keyword_boost and keywords:
            results = self._apply_keyword_boost(results, keywords)
            results = results[:k]  # Take top k after reranking
        
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

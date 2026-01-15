"""
Vector Store Module
Manages storage and retrieval of document embeddings.
Uses ChromaDB for local vector storage with hybrid search support.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Any, cast
from dataclasses import dataclass, asdict

import numpy as np
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""
    chunk_id: str
    content: str
    score: float
    source_file: str
    metadata: Optional[dict] = None


class VectorStore:
    """
    Manages document embeddings using ChromaDB.
    Supports dense retrieval with optional hybrid search.
    """
    
    def __init__(
        self,
        persist_directory: str = "data/vector_store",
        collection_name: str = "documents",
        embedding_dimension: int = 1024
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the ChromaDB collection
            embedding_dimension: Dimension of embeddings (1024 for BGE-M3)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"VectorStore initialized: {persist_directory}, collection: {collection_name}")
        logger.info(f"Current document count: {self.collection.count()}")
    
    def add_documents(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[np.ndarray],
        source_files: list[str],
        metadatas: Optional[list[dict]] = None
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            chunk_ids: Unique IDs for each chunk
            texts: Text content of each chunk
            embeddings: Dense embeddings for each chunk
            source_files: Source file names for each chunk
            metadatas: Optional metadata for each chunk
            
        Returns:
            Number of documents added
        """
        if not chunk_ids:
            return 0
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in chunk_ids]
        
        # Add source file to metadata
        for i, source_file in enumerate(source_files):
            metadatas[i]["source_file"] = source_file
        
        # Convert embeddings to list format
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        # Add to ChromaDB
        self.collection.add(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=cast(Any, metadatas)
        )
        
        logger.info(f"Added {len(chunk_ids)} documents to vector store")
        return len(chunk_ids)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_source: Optional[str] = None
    ) -> list[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_source: Optional source file filter
            
        Returns:
            List of SearchResult objects
        """
        # Prepare query
        query_emb_list = query_embedding.tolist()
        
        # Build filter if specified
        where_filter = None
        if filter_source:
            where_filter = {"source_file": filter_source}
        
        # Execute search
        results = self.collection.query(
            query_embeddings=[query_emb_list],
            n_results=top_k,
            where=cast(Any, where_filter),
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to SearchResult objects
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                # ChromaDB returns distances, convert to similarity scores
                distance = results['distances'][0][i] if results['distances'] else 0
                score = 1 - distance  # Convert distance to similarity
                
                docs = results['documents']
                metas = results['metadatas']
                content = docs[0][i] if docs else ''
                metadata = metas[0][i] if metas else {}
                source = str(metadata.get('source_file', 'unknown')) if metadata else 'unknown'
                
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    score=score,
                    source_file=source,
                    metadata=dict(metadata) if metadata else None
                ))
        
        logger.info(f"Search returned {len(search_results)} results")
        return search_results
    
    def delete_by_source(self, source_file: str) -> int:
        """
        Delete all documents from a specific source file.
        
        Args:
            source_file: Source file to delete documents for
            
        Returns:
            Number of documents deleted
        """
        # Get documents to delete
        results = self.collection.get(
            where={"source_file": source_file},
            include=["metadatas"]
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} documents from {source_file}")
            return len(results['ids'])
        
        return 0
    
    def get_all_sources(self) -> list[str]:
        """Get list of all unique source files in the store."""
        results = self.collection.get(include=["metadatas"])
        sources: set[str] = set()
        metadatas = results.get('metadatas') or []
        for metadata in metadatas:
            if metadata and 'source_file' in metadata:
                sources.add(str(metadata['source_file']))
        return list(sources)
    
    def count(self) -> int:
        """Return total number of documents in the store."""
        return self.collection.count()
    
    def reset(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Vector store reset")


if __name__ == "__main__":
    # Example usage
    store = VectorStore()
    
    # Example: Add some test documents
    test_ids = ["doc1", "doc2", "doc3"]
    test_texts = [
        "Machine learning is transforming industries.",
        "Financial markets show volatility.",
        "Climate change requires urgent action."
    ]
    # Create dummy embeddings (in practice, use EmbeddingGenerator)
    test_embeddings = [np.random.randn(1024) for _ in test_ids]
    test_sources = ["test.pdf"] * 3
    
    store.add_documents(test_ids, test_texts, test_embeddings, test_sources)
    
    print(f"Total documents: {store.count()}")
    print(f"Sources: {store.get_all_sources()}")

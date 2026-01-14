"""
Example usage of the RAG Pipeline.
Demonstrates document ingestion and querying.
"""

from src.rag_pipeline import RAGPipeline


def main():
    # Initialize the RAG pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(
        source_dir="data/source_pdfs",
        output_dir="data/processed_markdown",
        vector_store_dir="data/vector_store",
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5
    )
    
    # Check current stats
    stats = pipeline.get_stats()
    print(f"\nCurrent pipeline status:")
    print(f"  - Total documents in vector store: {stats['total_documents']}")
    print(f"  - Sources: {stats['sources']}")
    
    # Ingest documents if vector store is empty
    if stats['total_documents'] == 0:
        print("\n" + "="*50)
        print("Vector store is empty. Ingesting documents...")
        print("="*50)
        
        ingestion_stats = pipeline.ingest_documents()
        print(f"\nIngestion complete:")
        print(f"  - PDFs processed: {ingestion_stats['pdfs_processed']}")
        print(f"  - Chunks created: {ingestion_stats['chunks_created']}")
        print(f"  - Embeddings generated: {ingestion_stats['embeddings_generated']}")
        
        if ingestion_stats['errors']:
            print(f"  - Errors: {ingestion_stats['errors']}")
    
    # Example queries
    print("\n" + "="*50)
    print("Example Queries")
    print("="*50)
    
    sample_queries = [
        "What are the main findings of this document?",
        "Summarize the key financial metrics",
        "What recommendations are provided?"
    ]
    
    for query in sample_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        result = pipeline.query(query)
        
        if result.results:
            print(f"Found {len(result.results)} relevant chunks:")
            for i, res in enumerate(result.results[:3], 1):
                print(f"\n  [{i}] Score: {res.score:.3f} | Source: {res.source_file}")
                print(f"      {res.content[:150]}...")
        else:
            print("  No relevant documents found.")
    
    # Show how to get context for LLM
    print("\n" + "="*50)
    print("Context for LLM Integration")
    print("="*50)
    
    llm_prompt = pipeline.get_context_for_llm("What are the key takeaways?")
    print("\nFormatted prompt for LLM:")
    print(llm_prompt[:500] + "..." if len(llm_prompt) > 500 else llm_prompt)


if __name__ == "__main__":
    main()

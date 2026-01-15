"""
Example usage of the RAG Pipeline.
Demonstrates document ingestion and querying.
"""

import argparse
from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Example Usage")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocess all PDFs")
    parser.add_argument("--ask", type=str, help="Ask a question and get an answer using LLM")
    args = parser.parse_args()

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
    
    # Handle --ask argument (LLM-powered answers)
    if args.ask:
        print(f"\n📝 Question: {args.ask}")
        print("-" * 40)
        answer = pipeline.answer(args.ask)
        print(f"\n💡 Answer:\n{answer}")
        return
    
    # Ingest documents if vector store is empty OR --ingest flag is passed
    if stats['total_documents'] == 0 or args.ingest:
        print("\n" + "="*50)
        if args.force_reprocess:
            print("Force reprocessing all documents...")
        else:
            print("Ingesting documents...")
        print("="*50)
        
        ingestion_stats = pipeline.ingest_documents(force_reprocess=args.force_reprocess)
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

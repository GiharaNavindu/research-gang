from langchain_core.tools import tool
from retrieval.hybrid_search import HybridRAG
from retrieval.reranker import FlashReranker

# Initialize global instances to maintain connection pooling
rag_engine = HybridRAG()
reranker = FlashReranker()

@tool
def search_knowledge_base(query: str) -> str:
    """
    Use this tool to search the internal knowledge base for factual information,
    documentation, or specialized context to answer the user's query.
    Input should be a highly specific, optimized search string.
    """
    # 1. Execute Hybrid Search
    candidates = rag_engine.search(query, limit=10)
    
    # 2. Execute Reranking
    top_results = reranker.rerank(query, candidates, top_k=3)
    
    if not top_results:
        return "System Status: No relevant information found in the knowledge base."
        
    # 3. Format citations for the LLM to synthesize
    formatted_results = []
    for res in top_results:
        # Extract metadata for citation injection
        source = res.get("metadata", {}).get("source", "Unknown Document")
        formatted_results.append(f"[Source: {source}]\n{res['text']}")
        
    return "\n\n---\n\n".join(formatted_results)
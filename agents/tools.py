from langchain_core.tools import tool
from retrieval.hybrid_search import HybridRAG
from retrieval.reranker import FlashReranker

class LazyProxy:
    """A helper class to defer instantiation of heavy classes until they are accessed.
    This prevents lock conflicts in uvicorn's reload/multiprocessing environments.
    """
    def __init__(self, creator):
        self._creator = creator
        self._instance = None

    def _get_instance(self):
        if self._instance is None:
            self._instance = self._creator()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)

    def __setattr__(self, name, value):
        if name in ("_creator", "_instance"):
            self.__dict__[name] = value
        else:
            setattr(self._get_instance(), name, value)

# Initialize global instances lazily to avoid locking or loading models during master process import
rag_engine = LazyProxy(lambda: HybridRAG())
reranker = LazyProxy(lambda: FlashReranker())

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
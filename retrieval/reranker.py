from flashrank import Ranker, RerankRequest

class FlashReranker:
    """Ultra-lightweight cross-encoder for final result re-ordering."""
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        # The Ranker will download the model to cache_dir on first run.
        # This specific model provides an excellent speed-to-accuracy ratio.
        self.ranker = Ranker(model_name=model_name, cache_dir="./models_cache")

    def rerank(self, query: str, retrieved_docs: list[dict], top_k: int = 3) -> list[dict]:
        """
        Takes candidates from Qdrant, formats them for FlashRank,
        and applies cross-encoder scoring to find the absolute best matches.
        """
        if not retrieved_docs:
            return []

        # FlashRank expects a specific schema: list of dicts with 'id' and 'text' keys
        passages = []
        for idx, doc in enumerate(retrieved_docs):
            passages.append({
                "id": str(idx),
                "text": doc["text"],
                "metadata": doc["metadata"] 
            })
            
        rerank_request = RerankRequest(query=query, passages=passages)
        
        # Rerank returns the list sorted by the cross-encoder's score
        reranked_results = self.ranker.rerank(rerank_request)
        
        # Slice the highly-curated top results to feed to our LLM
        return reranked_results[:top_k]
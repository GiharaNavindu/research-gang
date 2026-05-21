import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from core.config import config

class HybridRAG:
    """Manages Dense + Sparse retrieval using Qdrant."""
    
    def __init__(self, collection_name: str = "production_rag"):
        self.collection_name = collection_name
        # Using local disk for the DB. In a true prod cloud environment, 
        # this would be an API URL to a managed Qdrant cluster.
        self.client = QdrantClient(path="./qdrant_data") 
        self.setup_collection()

    def setup_collection(self):
        """Initializes the collection with both dense and sparse vector spaces."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384,  # Dimension size for BAAI/bge-small-en-v1.5
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )

    def ingest_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """Embeds and uploads documents to both vector spaces."""
        # 1. Generate embeddings using our centralized config
        dense_embeddings = list(config.dense_embedding_model.embed(documents))
        sparse_embeddings = list(config.sparse_embedding_model.embed(documents))
        
        points = []
        for i, text in enumerate(documents):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_embeddings[i].tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_embeddings[i].indices.tolist(),
                            values=sparse_embeddings[i].values.tolist()
                        )
                    },
                    # Store the original text alongside metadata for easy retrieval
                    payload={**metadatas[i], "text": text}
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Executes Reciprocal Rank Fusion (RRF) across both index types."""
        # Embed the incoming query into both spaces
        dense_query = list(config.dense_embedding_model.embed([query]))[0]
        sparse_query = list(config.sparse_embedding_model.embed([query]))[0]

        # Qdrant's Query API allows us to fetch candidates from both indexes
        # and fuse them via RRF in a single database call.
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query.tolist(),
                    using="dense",
                    limit=limit * 2  # Oversample candidates for better fusion
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist()
                    ),
                    using="sparse",
                    limit=limit * 2
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )
        
        # Clean up the output to be agent-friendly
        return [
            {
                "text": point.payload["text"],
                "metadata": {k: v for k, v in point.payload.items() if k != "text"},
                "score": point.score
            }
            for point in results.points
        ]
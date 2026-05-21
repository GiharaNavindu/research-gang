import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from fastembed import TextEmbedding, SparseTextEmbedding

# Load environment variables from .env
load_dotenv()

class AIConfig:
    """Centralized configuration for Models and Embeddings."""
    
    def __init__(self):
        # 1. The Orchestration LLM (GPT-4o-mini is cost-effective and highly capable for agents)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.1, # Low temperature for grounded, factual RAG
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 2. Dense Embeddings (Captures semantic meaning - "Apple" near "Fruit")
        # We use fastembed to run BAAI/bge-small-en-v1.5 locally and fast
        self.dense_embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # 3. Sparse Embeddings (Captures exact keywords - BM25 equivalent)
        # SPLADE creates sparse vectors highlighting highly relevant exact terms
        self.sparse_embedding_model = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1")

# Instantiate a singleton config to import across the app
config = AIConfig()
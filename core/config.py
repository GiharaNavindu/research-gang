import os
from dotenv import load_dotenv
# 1. Update the import statement
from langchain_google_genai import ChatGoogleGenerativeAI
from fastembed import TextEmbedding, SparseTextEmbedding

load_dotenv()

class AIConfig:
    """Centralized configuration for Models and Embeddings."""
    
    def __init__(self):
        # 2. Swap out ChatOpenAI for ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            temperature=0.1, # Low temperature for grounded, factual RAG
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # 3. Embeddings remain exactly the same
        self.dense_embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.sparse_embedding_model = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1")

config = AIConfig()
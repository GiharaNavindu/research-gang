import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agents.graph import build_graph
from agents.tools import rag_engine

app = FastAPI(
    title="Research Engine API",
    description="Backend API for Qdrant Hybrid RAG & LangGraph Agent with Human Escalation",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Precompile the LangGraph application
graph_app = build_graph()

# Pydantic Schemas
class MessageItem(BaseModel):
    role: str = Field(..., description="user, assistant, system, or tool")
    content: str = Field(..., description="Text content of the message")
    source: Optional[str] = Field(None, description="Source document citation if available")

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's latest query")
    history: List[MessageItem] = Field(default=[], description="Message history for context")

class ChatStep(BaseModel):
    node: str
    message: str
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    response: str
    needs_human: bool
    steps: List[ChatStep]
    history: List[MessageItem]

class DocumentIngestRequest(BaseModel):
    text: str = Field(..., description="Content to embed and ingest")
    source: str = Field(..., description="Source identifier, e.g., paper.pdf")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Optional metadata fields")

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

class SearchResultItem(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float

@app.get("/api/status")
async def get_status():
    """Checks the health and configurations of the system."""
    try:
        # Check Qdrant collection status
        collection_exists = rag_engine.client.collection_exists(rag_engine.collection_name)
        
        info = {
            "status": "online",
            "qdrant": {
                "collection_name": rag_engine.collection_name,
                "exists": collection_exists,
                "path": "./qdrant_data"
            },
            "models": {
                "llm": "gemini-2.5-flash-lite",
                "dense_embeddings": "BAAI/bge-small-en-v1.5",
                "sparse_embeddings": "prithvida/Splade_PP_en_v1",
                "reranker": "ms-marco-MiniLM-L-12-v2"
            }
        }
        return info
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Submits user query to the LangGraph agent, runs the agentic pipeline, 
    and returns steps, final message, and updated message history.
    """
    try:
        # 1. Reconstruct LangChain Message format from history
        messages = []
        for msg in req.history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                messages.append(SystemMessage(content=msg.content))
        
        # Append the new user message
        messages.append(HumanMessage(content=req.message))
        
        # Initial State
        state = {
            "messages": messages,
            "needs_human": False
        }
        
        steps = []
        final_response = "No response generated."
        needs_human = False
        
        # Run graph and capture intermediate step outputs
        for event in graph_app.stream(state):
            for node_name, node_state in event.items():
                last_msg = node_state["messages"][-1]
                content = last_msg.content
                
                tool_calls = None
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    tool_calls = [
                        {"name": tc['name'], "args": tc['args'], "id": tc.get('id')}
                        for tc in last_msg.tool_calls
                    ]
                    content = f"Invoking tool: {last_msg.tool_calls[0]['name']}"
                
                steps.append(ChatStep(
                    node=node_name.upper(),
                    message=content,
                    tool_calls=tool_calls
                ))
                
                # Check human escalation flag
                if node_state.get("needs_human"):
                    needs_human = True
                    
                # Store the last response from orchestrator or human escalation
                if node_name in ["orchestrator", "human_escalation"]:
                    final_response = content

        # Update the API history representation to return to client
        updated_history = []
        # Include original history
        for m in req.history:
            updated_history.append(m)
        # Include new user message
        updated_history.append(MessageItem(role="user", content=req.message))
        # Include final assistant response
        role = "system" if needs_human else "assistant"
        updated_history.append(MessageItem(role=role, content=final_response))
        
        return ChatResponse(
            response=final_response,
            needs_human=needs_human,
            steps=steps,
            history=updated_history
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents")
async def ingest_document(req: DocumentIngestRequest):
    """
    Ingests a single text snippet or document metadata into the Hybrid Qdrant database.
    """
    try:
        documents = [req.text]
        metadatas = [{**req.metadata, "source": req.source}]
        
        # Ingest documents
        rag_engine.ingest_documents(documents=documents, metadatas=metadatas)
        return {"status": "success", "message": f"Successfully ingested document from '{req.source}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", response_model=List[SearchResultItem])
async def search_endpoint(req: SearchRequest):
    """
    Runs direct hybrid search and rerank. Useful for debugging and testing search accuracy.
    """
    try:
        from agents.tools import reranker
        
        # 1. Dense + Sparse Hybrid Search
        candidates = rag_engine.search(req.query, limit=req.limit * 2)
        
        # 2. FlashRank Rerank
        top_results = reranker.rerank(req.query, candidates, top_k=req.limit)
        
        results = []
        for r in top_results:
            results.append(SearchResultItem(
                text=r["text"],
                metadata=r.get("metadata", {}),
                score=r.get("score", 0.0)
            ))
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def list_documents(limit: int = 50):
    """
    Lists documents currently stored in the Qdrant vector database collection.
    """
    try:
        # Fetch points from Qdrant directly
        points = rag_engine.client.scroll(
            collection_name=rag_engine.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )[0]
        
        docs = []
        for p in points:
            docs.append({
                "id": str(p.id),
                "text": p.payload.get("text", ""),
                "source": p.payload.get("source", "Unknown"),
                "metadata": {k: v for k, v in p.payload.items() if k not in ["text", "source"]}
            })
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

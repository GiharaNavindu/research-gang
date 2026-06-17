# Research Engine (research-gang)

An autonomous technical assistant powered by LangGraph, Gemini 2.5, and a high-precision Hybrid RAG pipeline.

## Features
- **Agentic Workflow:** Uses LangGraph for stateful, cyclic agent logic.
- **Hybrid Retrieval:** Combines Dense (BGE) and Sparse (SPLADE) embeddings.
- **Advanced Fusion:** Implements Reciprocal Rank Fusion (RRF) natively in Qdrant.
- **Cross-Encoder Reranking:** Uses FlashRank for final result precision.
- **Human-in-the-Loop:** Built-in escalation path for complex queries.

## Setup Instructions

### 1. Prerequisites
- Python 3.10+
- A Google AI Studio API Key (for Gemini)

### 2. Installation
```bash
# Clone the repository (if applicable)
# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory and add your Google API key:
```env
GOOGLE_API_KEY=your_actual_api_key_here
```

## How to Run & Test

### 1. Start the System
Run the main script:
```bash
python main.py
```

### 2. Testing the End-to-End Flow
The system automatically ingests a few sample documents on startup to seed the knowledge base.

#### Test Case 1: Grounded Fact Retrieval
**Input:** `Why use cross-encoders?`
**Expected Behavior:** 
- The Orchestrator should call `search_knowledge_base`.
- The RAG pipeline will retrieve the sample doc: *"Production AI systems require cross-encoder reranking to minimize hallucination."*
- The LLM should provide an answer citing the source `AI_Architecture_Doc.pdf`.

#### Test Case 2: Multi-step Reasoning
**Input:** `Tell me about LangGraph and how it relates to hybrid search.`
**Expected Behavior:**
- The agent may search for both terms.
- It should synthesize information from both `LangGraph_Docs.md` and `Qdrant_Manual.txt`.

#### Test Case 3: Human Escalation
**Input:** `I need human help with this.`
**Expected Behavior:**
- The system should transition to the `human_escalation` node.
- It will output: `[SYSTEM]: Execution paused. Escalated to human operator.`

## Project Structure
- `main.py`: Entry point and CLI loop.
- `agents/`: Agent definitions and tool bindings.
- `retrieval/`: Hybrid search and reranking logic.
- `core/`: Global configuration (Models/Embeddings).
- `PROJECT_ANALYSIS.md`: Detailed architectural deep-dive.

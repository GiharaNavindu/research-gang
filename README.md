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

### Option A: The Full Web Application (Recommended)

We have implemented a beautiful, premium React + TypeScript + Tailwind CSS web interface with a FastAPI backend.

1. **Launch the entire system:**
   Double-click the **`run_system.bat`** file in the root directory, or execute it via terminal:
   ```bash
   .\run_system.bat
   ```
   This will automatically spin up:
   - **FastAPI Backend Server:** `http://127.0.0.1:8000`
   - **Vite React Web Frontend:** `http://127.0.0.1:5173`

2. **Web Interface Capabilities:**
   - **Agent Chat Dashboard:** Chat with the LangGraph agent in real-time. Review step-by-step intermediate execution traces, tool arguments, and cited source document cards.
   - **Human Escalation Node Control:** If the LLM pauses for human operator assistance (e.g. searching for human supervision or escalating), the UI presents an interactive overlay allowing you to input supervisor override instructions.
   - **Knowledge Ingestion Form:** Paste research text, add metadata tags and source names, and instantly embed/ingest documents into the local Qdrant collection.
   - **Direct Search Sandbox:** Query the hybrid vector search database directly to trace the dense/sparse embeddings score distribution before and after FlashRank cross-encoder reranking.

---

### Option B: The CLI Interactive Loop

If you prefer terminal-only execution:
```bash
python main.py
```

---

## Testing the End-to-End Flow

### Test Case 1: Grounded Fact Retrieval
- **Query:** `Why use cross-encoders?`
- **Behavior:** The agent invokes the hybrid search, fetches context from `AI_Architecture_Doc.pdf`, and cites the source automatically.

### Test Case 2: Human Escalation Loop
- **Query:** `I need human help with this`
- **Behavior:** The agent transitions to the `human_escalation` node.
  - In the CLI, the loop prints the pause status.
  - In the Web UI, an interactive supervisor override banner enables you to type override instructions to redirect the agent.

---

## Project Structure
- `server.py`: FastAPI server exposing the agent, ingest, status, and search APIs.
- `main.py`: CLI entry point and interactive loop.
- `run_system.bat`: One-click startup script for both servers.
- `agents/`: Stateful LangGraph node definitions, tool setups, and execution paths.
- `retrieval/`: Multi-stage RAG (Dense BGE + Sparse SPLADE + Reciprocal Rank Fusion + FlashRank Cross-Encoder).
- `core/`: Config values and AI model declarations.
- `frontend/`: React TypeScript single-page-app with Tailwind CSS styling and premium UI layouts.
- `PROJECT_ANALYSIS.md`: Detailed architectural analysis.
- `DEVELOPMENT_GUIDE.md`: Rebuilding steps overview.


import sys
from agents.graph import build_graph
from langchain_core.messages import HumanMessage
from agents.tools import rag_engine

def initialize_system():
    print("Initializing Vector Database and Ingesting Samples...")
    rag = rag_engine
    
    # Injecting sample data to test the RAG pipeline
    sample_docs = [
        "Production AI systems require cross-encoder reranking to minimize hallucination.",
        "LangGraph utilizes a state machine architecture for cyclic agent workflows.",
        "Qdrant supports Reciprocal Rank Fusion for hybrid search execution natively."
    ]
    sample_metadata = [
        {"source": "AI_Architecture_Doc.pdf"},
        {"source": "LangGraph_Docs.md"},
        {"source": "Qdrant_Manual.txt"}
    ]
    
    rag.ingest_documents(documents=sample_docs, metadatas=sample_metadata)
    print("Ingestion Complete.")

if __name__ == "__main__":
    initialize_system()
    
    # Compile the LangGraph application
    app = build_graph()
    print("\nSystem Online. Type 'exit' to quit.")
    print("Try asking: 'Why use cross-encoders?' or 'I need human help.'\n")
    
    while True:
        try:
            user_input = input("User> ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            # Initialize the state for this interaction
            state = {"messages": [HumanMessage(content=user_input)], "needs_human": False}
            
            # Stream the graph execution step-by-step
            for event in app.stream(state):
                for node_name, node_state in event.items():
                    print(f"\n--- Output from {node_name.upper()} ---")
                    
                    last_message = node_state["messages"][-1]
                    
                    # Print tool calls cleanly if they exist
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        print(f"Executing Tool: {last_message.tool_calls[0]['name']}")
                        print(f"Arguments: {last_message.tool_calls[0]['args']}")
                    else:
                        print(last_message.content)
                        
            print("\n" + "="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\nShutting down system.")
            sys.exit(0)
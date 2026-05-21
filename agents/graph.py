from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import AgentState
from agents.tools import search_knowledge_base
from core.config import config

# 1. Bind the RAG tool to the central LLM
tools = [search_knowledge_base]
llm_with_tools = config.llm.bind_tools(tools)

def orchestrator_node(state: AgentState):
    """The central decision engine."""
    messages = list(state["messages"])
    
    # Inject system instructions dynamically if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_prompt = SystemMessage(
            content="You are an autonomous technical assistant. "
                    "Use the search_knowledge_base tool to ground your answers in facts. "
                    "If the user explicitly requests human help, acknowledge it."
        )
        messages.insert(0, system_prompt)
        
    # Check if the user is explicitly requesting escalation
    last_user_msg = next((m.content.lower() for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    needs_human = "human" in last_user_msg or "escalate" in last_user_msg

    # Invoke the LLM (it will either generate text or a tool call)
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response], "needs_human": needs_human}

def human_escalation_node(state: AgentState):
    """Handles the hand-off to a human operator."""
    return {"messages": [SystemMessage(content="[SYSTEM]: Execution paused. Escalated to human operator.")]}

def routing_logic(state: AgentState):
    """Determines the next edge in the graph."""
    if state.get("needs_human"):
        return "human_escalation"
    
    last_message = state["messages"][-1]
    
    # If the LLM decided to invoke a tool, route to the tool node
    if last_message.tool_calls:
        return "execute_tools"
        
    # Otherwise, the LLM generated a final response
    return END

def build_graph():
    """Compiles the nodes and edges into a runnable state machine."""
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("execute_tools", ToolNode(tools)) # Prebuilt node that executes bound tools
    workflow.add_node("human_escalation", human_escalation_node)
    
    # Set the starting point
    workflow.set_entry_point("orchestrator")
    
    # Add Edges (Routing)
    workflow.add_conditional_edges(
        "orchestrator",
        routing_logic,
        {
            "execute_tools": "execute_tools",
            "human_escalation": "human_escalation",
            END: END
        }
    )
    
    # After tools execute, always return to the orchestrator to synthesize the results
    workflow.add_edge("execute_tools", "orchestrator")
    
    # End the graph after human escalation
    workflow.add_edge("human_escalation", END)
    
    return workflow.compile()
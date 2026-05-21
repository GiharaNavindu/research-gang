from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    The core state payload passed between nodes in our graph.
    """
    # The 'add_messages' annotation ensures that when a node returns a new message, 
    # it is appended to the existing sequence rather than overwriting it.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # A flag to pause execution and route to a human supervisor
    needs_human: bool
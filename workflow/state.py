from typing import Annotated, List, Dict, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from tools import ToolCall
from tools.apis import AgentContext


class AgentState(TypedDict):
    """
    The complete, structured state for the agent, managed by LangGraph.
    
    Attributes:
        session_id (Optional[str]): A unique identifier for the entire session, used
            for logging, visualization, and output management.
        messages (Annotated[...]): The historical log of the interaction, including 
            human inputs, AI-generated plans and tool results, used for LLM planner 
            and debugging.
        workspace (Dict[str, AgentContext]): The computable workspace where all named 
            variables (tool outputs and initial inputs) are stored.
            Note: The output of MetaPlanner ('task_router_output') is also stored here.
        current_plan (List[ToolCall]): The actionable plan for the next step, generated
            by the AgentPlanner. This list of ToolCall objects is consumed by the 
            AgentToolExecutor.
        variable_to_call_id_map (Dict[str, str]): A lookup map for data provenance,
            tracking the lineage of variables. It maps an output variable name to the
            unique `call_id` of the tool that successfully created it. 
        call_id_to_input_map (Dict[str, List[str]]): It maps a `call_id` to the list 
            of input variable names it consumed from the workspace.
    """
    session_id: Optional[str]

    messages: Annotated[List[AnyMessage], add_messages]
    workspace: Dict[str, AgentContext]
    current_plan: List[ToolCall]

    variable_to_call_id_map: Dict[str, str]
    call_id_to_input_map: Dict[str, List[str]]

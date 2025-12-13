from langgraph.graph import END

from workflow.state import AgentState


def after_solver_planning(state: AgentState) -> str:
    if state.get('current_plan', []):
        return 'solver_executor'
    return END


def after_solver_execution(state: AgentState) -> str:
    if 'final_answer' in state.get('workspace', {}):
        return END
    
    return 'solver_planner'


def after_meta_planning(state: AgentState) -> str:
    workspace = state.get('workspace', {})
    router_output = workspace.get('task_router_output')
    
    decision = 'image-driven'  # default
    if router_output:
        decision = router_output.decision
        
    if decision == 'image-driven':
        return 'semantic_analyst'
    elif decision == 'text-driven':
        return 'solver_planner'
    
    return 'semantic_analyst'

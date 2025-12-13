import json
from typing import Dict, List, Tuple

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage


# Prevent circular import
def parse_call_id(call_id: str) -> Tuple[str, str, str]:
    parts = call_id.split('/')
    assert len(parts) >= 3
    step_id, tool_name, output_variable = parts[0], parts[1], '/'.join(parts[2:])
    return step_id, tool_name, output_variable


def format_ai_message(message: AIMessage, turn_counter: int) -> str:
    msg_type = message.additional_kwargs.get('_type', 'default')

    if msg_type == 'solver_planning':
        content = message.content[0]
        if isinstance(content, Dict):
            analysis = (
                f'**Current Situation**: {content["current_situation"]}\n'
                f'**Next Plan**: {content["next_plan"]}'
            )
        else:
            analysis = content
        return f'|TURN {turn_counter} - ANALYSIS|\n{analysis}'
    
    if msg_type == 'ref_frame_constraint':
        content = (
            f'**Formalization**: {message.content}\n'
            f'**Reasoning**: {message.additional_kwargs["reasoning_content"]}'
        )
        return f'|TASK FORMALIZATION - REFERENCE FRAME|\n{content}'
    
    if msg_type == 'objective_constraint':
        content = (
            f'**Objective**: {message.content}\n'
            f'**Reasoning**: {message.additional_kwargs["reasoning_content"]}'
        )
        return f'|TASK FORMALIZATION - OBJECTIVE|\n{content}'
    
    # Default case: return the message content as-is
    if msg_type == 'default':
        return f'|TURN {turn_counter} - AI RESPONSE|\n{message.content}'
    
    raise KeyError(f'Unsupported AIMessage type {msg_type}.')


def format_tool_message(message: ToolMessage) -> str:
    call_id = message.tool_call_id.strip()
    step_id, tool_name, output_variable = parse_call_id(call_id)

    tool_call_data = {
        'tool_name': tool_name,
        'status': message.status,
        'output_variable': output_variable,
        'args': message.additional_kwargs['args'],
        'output_message': message.content
    }

    message = '- ' + '\n'.join([
        '  ' + line
        for line in json.dumps(tool_call_data, indent=2).split('\n')
    ])[2:]

    return message


def format_history_messages(messages: List[AnyMessage]) -> str:
    if not messages:
        return 'No history yet.'

    history = []
    i, turn_counter = 0, 0

    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            if i == 0:
                history.append(f'|USER\'S GOAL|\n{msg.content}')
            # hide intermediate error prompt to avoid corrupt context
            elif i == len(messages) - 1 and msg.additional_kwargs.get('retry', -1) > 0:
                history.append(
                    f'|SYSTEM CORRECTION|\n{msg.content}\nError Output Last Time:\n'
                    f'{msg.additional_kwargs["content"]}'
                )
            i += 1
        elif isinstance(msg, AIMessage):
            turn_counter += 1
            history.append(format_ai_message(msg, turn_counter))
            i += 1
        elif isinstance(msg, ToolMessage):
            tool_message_group = []
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                tool_message_group.append(messages[i])
                i += 1

            if tool_message_group:
                formatted_tool_calls = []
                for tool_msg in tool_message_group:
                    formatted_tool_calls.append(format_tool_message(tool_msg))

                tool_call_history = f'|TURN {turn_counter} - TOOL CALL RESULTS|\n' + \
                    '\n'.join(formatted_tool_calls)
                history.append(tool_call_history)
    
    return '\n\n'.join(history)

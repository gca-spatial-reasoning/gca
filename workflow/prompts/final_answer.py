from typing import Any, List

from langchain_core.messages import AnyMessage

from tools.apis import AgentContext
from workflow.utils.msg_utils import format_history_messages


FINAL_ANSWER_PROMPT_V1 = """
The agent has completed its task. Below is the complete history of the user's request and the agent's actions:

---- CONVERSATION HISTORY ----
{history_str}

---- EXECUTION RESULT ----
{answer_str}

Based on the conversation history and execution result, provide a natural language answer to the user's original request.
**Crucial Rules:** 
- If there is a conflict between planner's analysis and the execution result, please take the execution result as the final answer.
- The final answer must correspond exactly to the question; for example, for a multiple-choice question, the answer must be the specific option.
- Please adhere to the format required by the original request (if specified).
""".strip()


def build_final_answer_prompt(
    final_answer: AgentContext | Any,
    messages: List[AnyMessage],
) -> str:
    history_str = format_history_messages(messages)
    user_request = messages[0].content

    if isinstance(final_answer, AgentContext):
        answer_type = type(final_answer)
        answer_summary = final_answer.to_message_content()
        type_doc = getattr(answer_type, '__doc__')
        answer_str = f"""
The final result is an object of type `{answer_type.__name__}`, which 
{type_doc.lstrip()[0].lower()}{type_doc.lstrip()[1:]}

- Summary: {answer_summary}
- Full Value: {final_answer}
""".strip()

    else:
        # The final result is a `{type(final_answer).__name__}` with the value:
        answer_str = f"{str(final_answer)}"

    return FINAL_ANSWER_PROMPT_V1.format(
        history_str=history_str,
        answer_str=answer_str,
    )

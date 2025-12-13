import asyncio
from typing import Any, Callable, Optional


async def invoke_with_retry(
    invoker: Callable,
    prompter: Callable,
    parser: Optional[Callable],
    max_retries: int = 3
) -> Any:
    err_msg, response_content = None, None

    for i in range(max_retries + 1):
        try:
            prompt = prompter(err_msg=err_msg, response=response_content)
            output = await invoker(prompt=prompt)
            if output.err:
                raise RuntimeError(output.err['msg'])
            
            response_content = output.result.content
            if parser is not None:
                return await parser(output.result)
            else:
                return response_content

        except Exception as e:
            err_msg = str(e)
            if i < max_retries:
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError(err_msg)

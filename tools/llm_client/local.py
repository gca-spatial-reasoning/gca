from typing import Any, Dict, List

from ray.serve.handle import DeploymentHandle
from openai.types.chat import ChatCompletion


class LocalChatCompletions:
    def __init__(self, handle: DeploymentHandle):
        self.handle = handle

    async def create(self, **kwargs) -> ChatCompletion:
        result_dict = await self.handle.generate.remote(**kwargs)
        return ChatCompletion.model_validate(result_dict)


class LocalChat:
    def __init__(self, handle: DeploymentHandle):
        self.completions = LocalChatCompletions(handle)


class AsyncLocalModelClient:
    def __init__(self, handle: DeploymentHandle):
        self.chat = LocalChat(handle)

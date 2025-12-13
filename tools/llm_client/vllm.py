import random
from typing import List

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion


class VLLMLBChatCompletions:
    def __init__(
        self, 
        endpoints: List[str], 
        api_key: str, 
        proxy_manager, 
        client_name: str
    ):
        self.endpoints = endpoints
        self.api_key = api_key
        self.proxy_manager = proxy_manager
        self.client_name = client_name

    async def create(self, **kwargs) -> ChatCompletion:
        base_url = random.choice(self.endpoints)
        httpx_client = self.proxy_manager.get_httpx_client(self.client_name)
        
        async with AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            http_client=httpx_client,
        ) as client:
            response = await client.chat.completions.create(**kwargs)
        
        return response


class VLLMLBChat:
    def __init__(
        self, 
        endpoints: List[str], 
        api_key: str, 
        proxy_manager, 
        client_name: str
    ):
        self.completions = VLLMLBChatCompletions(
            endpoints, api_key, proxy_manager, client_name
        )
    

class AsyncVLLMLBClient:
    def __init__(
        self, 
        endpoints: List[str], 
        api_key: str, 
        proxy_manager, 
        client_name: str
    ):
        self.chat = VLLMLBChat(
            endpoints, api_key, proxy_manager, client_name
        )

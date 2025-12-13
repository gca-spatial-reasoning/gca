import json
import os
from typing import List, Tuple, Union

from openai import AsyncOpenAI
from ray import serve

from tools.llm_client.local import AsyncLocalModelClient
from tools.llm_client.local_deployment import LOCAL_DEPLOYMENTS
from tools.llm_client.vllm import AsyncVLLMLBClient
from tools.utils.proxy_manager import ProxyManager
from workflow.config import get_config


AsyncClient = Union[
    AsyncOpenAI,
    AsyncVLLMLBClient,
    AsyncLocalModelClient
]


class LLMClientFactory:

    def __init__(self):
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        self.serve_file = os.path.join(log_dir, 'serve.json')
        self.proxy_manager = ProxyManager()

    def get_all_vllm_endpoints(self, model: str) -> List[str]:
        if not os.path.exists(self.serve_file):
            raise FileNotFoundError(
                f'serve.json not found at "{self.serve_filel}". Is the vLLM service running?'
            )
        
        try:
            with open(self.serve_file, 'r', encoding='utf-8') as f:
                serve_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            raise ValueError(f'Could not read or parse serve.json at "{self.serve_file}".')
        
        model_instances = serve_data.get(model)
        if not model_instances or len(model_instances) < 1:
            raise ValueError(f'No running service found for model "{model}" in serve.json.')
        
        endpoints = []
        for instance in model_instances.values():
            ip, port = instance['ip'], instance['port']
            endpoints.append(f'http://{ip}:{port}/v1')
        return endpoints
    
    def create_vllm_client(
        self,
        model: str,
        client_name: str,
    ) -> Tuple[AsyncVLLMLBClient, str]:
        all_endpoints = self.get_all_vllm_endpoints(model)
        print(f'Find {len(all_endpoints)} instances for {model}.')
        client = AsyncVLLMLBClient(
            endpoints=all_endpoints,
            api_key='bearer',
            proxy_manager=self.proxy_manager,
            client_name=client_name,
        )
        return client, model

    def create_local_model_client(
        self,
        model: str,
    ) -> Tuple[AsyncLocalModelClient, str]:
        model_name = model.lower()
        if model_name not in LOCAL_DEPLOYMENTS:
            valid_keys = list(LOCAL_DEPLOYMENTS.keys())
            raise ValueError(
                f'Local model {model_name} not supported. Expected {valid_keys}'
            )
        
        try:
            handle = serve.get_app_handle(model_name)
        except Exception:
            model_class = LOCAL_DEPLOYMENTS[model_name]
            if isinstance(model_class, serve.Deployment):
                model_application = model_class.bind()
            elif isinstance(model_class, serve.Application):
                model_application = model_class

            serve.run(
                model_application,
                name=model_name,
                route_prefix=f'/{model_name}'
            )
            handle = serve.get_app_handle(model_name)
        
        client = AsyncLocalModelClient(handle)
        return client, model

    def create_default_client(
        self,
        model: str,
        base_url: str,
        api_key: str,
        client_name: str,
    ) -> Tuple[AsyncOpenAI, str]:
        if api_key is None:
            raise ValueError(f'Environment variable for api key not set.')
        
        httpx_client = self.proxy_manager.get_httpx_client(client_name)
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx_client,
        )
        return client, model
    
    def create_client(
        self,
        client_name: str,
    ) -> Tuple[AsyncClient, str]:
        config = get_config()
        model = getattr(config, f'{client_name}_model')
        base_url = getattr(config, f'{client_name}_base_url')
        api_key = getattr(config, f'{client_name}_api_key')

        if model is None:
            raise ValueError(f'"{client_name}_model" not set in Config.')
        if base_url is None:
            raise ValueError(f'"{client_name}_base_url" not set in Config.')
        
        if base_url.lower().strip() == 'vllm':
            return self.create_vllm_client(model, client_name)
        if base_url.lower().strip() == 'local':
            return self.create_local_model_client(model)
        return self.create_default_client(model, base_url, api_key, client_name)

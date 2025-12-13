from typing import Optional

import httplib2
import httpx

from workflow.config import get_config


class ProxyManager:

    def __init__(self):
        config = get_config()
        self.proxies = {
            'cot_reasoner': config.cot_reasoner_proxy,
            'code_generator': config.code_generator_proxy,
            'img_loader': config.img_loader_proxy,
        }

    def get_proxy(self, key: str) -> str:
        return self.proxies.get(key, '')
    
    def create_httpx_client(self, proxy_url: str) -> httpx.AsyncClient:
        proxies = {'http://': proxy_url, 'https://': proxy_url}
        return httpx.AsyncClient(proxies=proxies, http2=True)

    def get_httpx_client(self, key: str) -> Optional[httpx.AsyncClient]:
        proxy_url = self.proxies.get(key, '')
        if not proxy_url:
            return None
        return self.create_httpx_client(proxy_url)

    def get_google_api_http_client(self) -> Optional[httplib2.Http]:
        proxy_url = self.proxies.get('google_api', '')
        if not proxy_url:
            return None

        from urllib.parse import urlparse
        parse_result = urlparse(proxy_url)
        proxy_type = parse_result.scheme.lower()
        if proxy_type not in ["http", "httpss", "socks4", "socks5"]:
            raise ValueError(f"Unsupported proxy type for Google API: {proxy_type}")

        proxy_info = httplib2.ProxyInfo(
            proxy_type=getattr(httplib2.socks, f'PROXY_TYPE_{proxy_type.upper()}'),
            proxy_host=parse_result.hostname,
            proxy_port=parse_result.port,
        )
        return httplib2.Http(proxy_info=proxy_info)

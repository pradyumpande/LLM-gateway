from abc import ABC, abstractmethod
from typing import Any

import httpx


class ProviderError(Exception):
    def __init__(self, status_code: int, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.retryable = retryable


class BaseProvider(ABC):
    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key: str,
        client: httpx.AsyncClient,
    ) -> None:
        self.base_url = base_url
        self.model_id = model_id
        self.api_key = api_key
        self.client = client

    @abstractmethod
    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

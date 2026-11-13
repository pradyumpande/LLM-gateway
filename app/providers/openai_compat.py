from typing import Any

import httpx

from app.providers.base import BaseProvider, ProviderError


class OpenAICompatProvider(BaseProvider):
    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload = {**payload, "model": self.model_id}
        url = f"{self.base_url.rstrip('/')}/chat/completions"

        try:
            response = await self.client.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_payload,
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                504,
                "Provider request timed out.",
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError(
                502,
                "Could not connect to the provider.",
                retryable=True,
            ) from exc

        if not response.is_success:
            status_code = response.status_code
            raise ProviderError(
                status_code,
                self._error_message(response),
                retryable=status_code == 429 or status_code >= 500,
            )

        return response.json()

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = None

        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            elif isinstance(error, str) and error:
                return error

            message = body.get("message")
            if isinstance(message, str) and message:
                return message

        return f"Provider returned HTTP {response.status_code}."
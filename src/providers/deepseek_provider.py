"""DeepSeek provider implementation."""

from __future__ import annotations

from typing import Any, Optional

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

from .openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider using OpenAI-compatible API."""

    def __init__(
        self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None
    ):
        super().__init__(
            api_key,
            base_url or "https://api.deepseek.com",
            model or "deepseek-v4-pro",
        )

    def _create_client(self) -> Any:
        if OpenAI is None:
            raise ModuleNotFoundError(
                "openai package is not installed. Install optional dependencies to use DeepSeekProvider."
            )
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_available_models(self) -> list[str]:
        return [
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "deepseek-chat",
            "deepseek-coder",
            "deepseek-reasoner",
        ]

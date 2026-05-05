"""Qwen (Alibaba Tongyi Qianwen) provider implementation."""

from __future__ import annotations

from typing import Any, Optional

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

from .openai_compatible import OpenAICompatibleProvider


class QwenProvider(OpenAICompatibleProvider):
    """Qwen provider using OpenAI-compatible API."""

    def __init__(
        self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None
    ):
        super().__init__(
            api_key,
            base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model or "qwen3.6-plus",
        )

    def _create_client(self) -> Any:
        if OpenAI is None:
            raise ModuleNotFoundError(
                "openai package is not installed. Install optional dependencies to use QwenProvider."
            )
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_available_models(self) -> list[str]:
        return [
            "qwen3.6-plus",
            "qwen3.6-plus-2026-04-02",
            "qwen3.6-35b-a3b",
            "qwen3.6-flash-2026-04-16",
            "qwen3.5-flash",
            "qwen3.5-35b-a3b",
            "qwen3-coder-next",
            "qwen-plus",
            "qwen-turbo",
            "qwen-max",
            "qwen-max-longcontext",
            "qwen-long",
            "qwen2.5-72b-instruct",
            "qwen2.5-32b-instruct",
            "qwen2.5-14b-instruct",
            "qwen2.5-7b-instruct",
        ]

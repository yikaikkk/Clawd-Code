"""Mem0-backed long-term memory service."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any


def _project_id(workspace_root: Path) -> str:
    root = str(workspace_root.expanduser().resolve())
    digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:16]
    return f"workspace:{digest}"


def _env_value(config: dict[str, Any]) -> str:
    env_name = config.get("api_key_env")
    if isinstance(env_name, str) and env_name:
        value = os.environ.get(env_name, "")
        if value:
            return value
        if _looks_like_api_key(env_name):
            return env_name
    api_key = config.get("api_key")
    return api_key if isinstance(api_key, str) else ""


def _strip_empty(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v not in ("", None)}


def _looks_like_api_key(value: str) -> bool:
    return value.startswith(("sk-", "sk_", "ak-", "ak_"))


# 默认的长期记忆关键词（用户画像和长期目标相关）
_DEFAULT_LONG_TERM_KEYWORDS = frozenset([
    # 用户画像相关
    "我是", "我的名字", "我叫", "我来自", "我从事", "我在", "我是做",
    "我的职业", "我的工作", "我的专业", "我的学历", "我的背景",
    "我擅长", "我熟悉", "我精通", "我了解", "我学习过",
    "我的目标", "我的计划", "我想", "我希望", "我打算",
    "我的需求", "我的要求", "我需要", "我想要", "我的偏好",
    "我的习惯", "我的风格", "我喜欢", "我不喜欢",
    # 长期目标相关
    "长期", "计划", "规划", "目标", "愿景", "战略", "方向",
    "未来", "以后", "将来", "接下来", "准备", "打算",
    "项目", "任务", "里程碑", "阶段", "步骤",
])


# 默认的信息提炼prompt
_DEFAULT_EXTRACTION_PROMPT = """
请从以下对话中提取并总结用户画像和长期目标相关的核心信息：

用户输入: {user_input}
助手回复: {assistant_output}

请以简洁、结构化的方式输出，只保留关键信息，格式如下：

用户画像: 
- 身份背景（职业、学历、专业等）
- 技能专长
- 偏好习惯

长期目标:
- 短期计划（1-3个月）
- 中期目标（3-12个月）
- 长期愿景（1年以上）

如果没有相关信息，请输出"无"。
"""


def _contains_long_term_keywords(text: str, keywords: frozenset[str]) -> bool:
    """检查文本是否包含长期记忆相关的关键词。"""
    if not isinstance(text, str) or not text.strip():
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in {"api_key", "password", "token", "secret"}:
                redacted[key] = "<redacted>" if item else item
            else:
                redacted[key] = _redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def format_memories_for_prompt(memories: list[dict[str, Any]]) -> str:
    """Format mem0 search results as a compact system-prompt section."""
    lines: list[str] = []
    seen: set[str] = set()
    for item in memories:
        text = item.get("memory") or item.get("text") or item.get("content")
        if not isinstance(text, str):
            continue
        text = " ".join(text.split())
        if not text or text in seen:
            continue
        seen.add(text)
        lines.append(f"- {text}")
    if not lines:
        return ""
    return (
        "Relevant long-term memory:\n"
        + "\n".join(lines)
        + "\n\nUse these memories only when relevant. Do not mention memory retrieval unless asked."
    )


class MemoryService:
    """Lazy wrapper around local Mem0 OSS.

    The service is deliberately fail-soft: memory should improve answers, never
    block normal chat when Chroma, Neo4j, or model credentials are unavailable.
    """

    def __init__(self, config: dict[str, Any] | None, workspace_root: Path, session_id: str | None = None):
        self.config = config or {}
        self.workspace_root = Path(workspace_root)
        self.session_id = session_id
        self.user_id = str(self.config.get("user_id") or "default")
        self.agent_id = str(self.config.get("agent_id") or "clawd-code")
        self.project_id = _project_id(self.workspace_root)
        self._client: Any | None = None
        self._disabled_reason: str | None = None
        self._debug(
            "initialized",
            enabled=self.enabled,
            inject=bool(self.config.get("inject", True)),
            write_after_turn=bool(self.config.get("write_after_turn", True)),
            workspace_root=str(self.workspace_root),
            session_id=self.session_id or "",
        )

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled"))

    @property
    def debug_enabled(self) -> bool:
        return bool(self.config.get("debug"))

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def _debug(self, message: str, **fields: Any) -> None:
        if not self.debug_enabled:
            return
        details = " ".join(
            f"{key}={_redact_sensitive(value)!r}" for key, value in fields.items()
        )
        suffix = f" {details}" if details else ""
        print(f"[clawd:memory] {message}{suffix}", file=sys.stderr)

    def _debug_error(self, operation: str, exc: Exception | str) -> None:
        self._debug(f"{operation} error", error=str(exc))

    def search_prompt(self, query: str) -> str:
        if not self.enabled:
            self._debug("search skipped", reason="memory disabled")
            return ""
        if not bool(self.config.get("inject", True)):
            self._debug("search skipped", reason="memory injection disabled")
            return ""
        memories = self.search(query)
        prompt = format_memories_for_prompt(memories)
        self._debug(
            "search prompt built",
            memories=len(memories),
            injected=bool(prompt),
        )
        return prompt

    def search(self, query: str) -> list[dict[str, Any]]:
        if not self.enabled:
            self._debug("search skipped", reason="memory disabled")
            return []
        if not query.strip():
            self._debug("search skipped", reason="empty query")
            return []
        client = self._get_client()
        if client is None:
            self._debug("search skipped", reason=self._disabled_reason or "client unavailable")
            return []
        try:
            limit = int(self.config.get("search_limit") or 5)
            filters = {"user_id": self.user_id}
            self._debug(
                "search started",
                query_chars=len(query),
                limit=limit,
                user_id=self.user_id,
                enable_graph=bool(self.config.get("enable_graph", True)),
            )
            try:
                result = client.search(
                    query=query,
                    limit=limit,
                    filters=filters,
                    enable_graph=bool(self.config.get("enable_graph", True)),
                )
            except TypeError:
                result = client.search(query=query, limit=limit, filters=filters)
            memories = self._normalize_results(result)
            self._debug("search succeeded", memories=len(memories))
            return memories
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._debug_error("search", exc)
            return []

    def _should_save_memory(self, user_input: str, assistant_output: str) -> bool:
        """判断是否应该保存这段对话到长期记忆。
        
        保存策略由配置项 `save_strategy` 控制：
        - "all": 保存所有对话（默认）
        - "long_term_only": 只保存包含用户画像和长期目标的内容
        """
        strategy = self.config.get("save_strategy", "all")
        
        if strategy == "all":
            return True
        
        if strategy == "long_term_only":
            # 获取自定义关键词列表
            custom_keywords = self.config.get("long_term_keywords")
            if isinstance(custom_keywords, list):
                keywords = frozenset(k.lower() for k in custom_keywords if isinstance(k, str))
            else:
                keywords = _DEFAULT_LONG_TERM_KEYWORDS
            
            # 检查用户输入或助手回复是否包含长期记忆关键词
            if _contains_long_term_keywords(user_input, keywords):
                self._debug("memory qualifies for long-term", reason="user input contains long-term keywords")
                return True
            if _contains_long_term_keywords(assistant_output, keywords):
                self._debug("memory qualifies for long-term", reason="assistant output contains long-term keywords")
                return True
            
            self._debug("write skipped", reason="does not contain long-term keywords")
            return False
        
        # 未知策略，默认保存
        return True

    def _extract_core_info(self, user_input: str, assistant_output: str) -> tuple[str, str]:
        """使用LLM提炼对话中的核心信息（用户画像和长期目标）。
        
        Args:
            user_input: 用户输入
            assistant_output: 助手回复
            
        Returns:
            提炼后的用户输入和助手回复（如果启用提炼）
        """
        # 检查是否启用提炼功能
        if not bool(self.config.get("enable_extraction", False)):
            return user_input, assistant_output
        
        # 获取提炼配置
        extraction_prompt = self.config.get("extraction_prompt")
        if not isinstance(extraction_prompt, str) or not extraction_prompt.strip():
            extraction_prompt = _DEFAULT_EXTRACTION_PROMPT
        
        # 获取provider配置
        provider_config = self.config.get("extraction_provider")
        if not isinstance(provider_config, dict):
            self._debug("extraction skipped", reason="no extraction_provider config")
            return user_input, assistant_output
        
        try:
            from src.providers import get_provider_class
            
            provider_name = provider_config.get("provider")
            if not isinstance(provider_name, str):
                self._debug("extraction skipped", reason="no provider name specified")
                return user_input, assistant_output
            
            provider_class = get_provider_class(provider_name)
            
            api_key = _env_value(provider_config)
            base_url = provider_config.get("base_url")
            model = provider_config.get("model")
            
            provider = provider_class(api_key=api_key, base_url=base_url, model=model)
            
            # 构建提炼prompt
            prompt = extraction_prompt.format(
                user_input=user_input,
                assistant_output=assistant_output
            )
            
            self._debug("extraction started", prompt_chars=len(prompt))
            
            # 调用LLM进行提炼
            response = provider.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            extracted_content = response.content.strip()
            
            # 如果提炼结果有效，替换原始内容
            if extracted_content and extracted_content != "无":
                self._debug("extraction succeeded", extracted_chars=len(extracted_content))
                return extracted_content, extracted_content
            else:
                self._debug("extraction skipped", reason="no useful info extracted")
                return user_input, assistant_output
                
        except Exception as exc:
            self._debug_error("extraction", exc)
            return user_input, assistant_output

    def add_turn(self, user_input: str, assistant_output: str) -> None:
        if not self.enabled:
            self._debug("write skipped", reason="memory disabled")
            return
        if not bool(self.config.get("write_after_turn", True)):
            self._debug("write skipped", reason="memory writes disabled")
            return
        if not user_input.strip() or not assistant_output.strip():
            self._debug("write skipped", reason="empty turn")
            return
        
        # 检查是否应该保存这段记忆
        if not self._should_save_memory(user_input, assistant_output):
            return
        
        # 提炼核心信息（如果启用）
        extracted_input, extracted_output = self._extract_core_info(user_input, assistant_output)
        
        client = self._get_client()
        if client is None:
            self._debug("write skipped", reason=self._disabled_reason or "client unavailable")
            return
        messages = [
            {"role": "user", "content": extracted_input},
            {"role": "assistant", "content": extracted_output},
        ]
        metadata = {
            "workspace_root": str(self.workspace_root),
            "project_id": self.project_id,
            "session_id": self.session_id or "",
            "memory_type": "long_term",
        }
        try:
            self._debug(
                "write started",
                user_chars=len(user_input),
                assistant_chars=len(assistant_output),
                user_id=self.user_id,
                agent_id=self.agent_id,
                project_id=self.project_id,
                enable_graph=bool(self.config.get("enable_graph", True)),
            )
            try:
                client.add(
                    messages,
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    metadata=metadata,
                    enable_graph=bool(self.config.get("enable_graph", True)),
                )
            except TypeError:
                client.add(messages, user_id=self.user_id, agent_id=self.agent_id, metadata=metadata)
            self._debug("write succeeded")
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._debug_error("write", exc)

    def _get_client(self) -> Any | None:
        if self._client is not None:
            self._debug("client reused")
            return self._client
        if not self.enabled:
            self._debug("client not initialized", reason="memory disabled")
            return None
        try:
            from mem0 import Memory  # type: ignore
        except Exception as exc:
            self._disabled_reason = f"mem0 import failed: {exc}"
            self._debug("client import failed", error=self._disabled_reason)
            return None
        try:
            mem0_config = self._build_mem0_config()
            self._debug("client initialization started", config=_redact_sensitive(mem0_config))
            try:
                self._client = Memory.from_config(config_dict=mem0_config)
            except TypeError:
                self._client = Memory.from_config(mem0_config)
            self._debug("client initialization succeeded")
            return self._client
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._debug("client initialization failed", error=self._disabled_reason)
            return None

    def _build_mem0_config(self) -> dict[str, Any]:
        chroma = self.config.get("chroma") if isinstance(self.config.get("chroma"), dict) else {}
        neo4j = self.config.get("neo4j") if isinstance(self.config.get("neo4j"), dict) else {}
        llm = self.config.get("llm") if isinstance(self.config.get("llm"), dict) else {}
        embedder = self.config.get("embedder") if isinstance(self.config.get("embedder"), dict) else {}

        chroma_config: dict[str, Any] = {
            "collection_name": chroma.get("collection_name") or "clawd_code_memory",
        }
        if chroma.get("host"):
            chroma_config["host"] = chroma.get("host")
            if chroma.get("port") is not None:
                chroma_config["port"] = chroma.get("port")
        else:
            chroma_config["path"] = str(Path(str(chroma.get("path") or "~/.clawd/memory/chroma")).expanduser())

        mem0_config: dict[str, Any] = {
            "vector_store": {
                "provider": "chroma",
                "config": chroma_config,
            },
            "version": "v1.1",
        }

        llm_config = _strip_empty({
            "model": llm.get("model"),
            "temperature": llm.get("temperature"),
            "api_key": _env_value(llm),
            "openai_base_url": llm.get("openai_base_url"),
        })
        if llm.get("provider"):
            mem0_config["llm"] = {"provider": llm.get("provider"), "config": llm_config}

        embedder_config = _strip_empty({
            "model": embedder.get("model"),
            "api_key": _env_value(embedder),
            "openai_base_url": embedder.get("openai_base_url"),
        })
        if embedder.get("provider"):
            mem0_config["embedder"] = {"provider": embedder.get("provider"), "config": embedder_config}

        if bool(self.config.get("enable_graph", True)):
            mem0_config["graph_store"] = {
                "provider": "neo4j",
                "config": _strip_empty({
                    "url": neo4j.get("url"),
                    "username": neo4j.get("username"),
                    "password": neo4j.get("password"),
                    "database": neo4j.get("database"),
                    "threshold": neo4j.get("threshold"),
                }),
            }

        return mem0_config

    @staticmethod
    def _normalize_results(result: Any) -> list[dict[str, Any]]:
        if isinstance(result, dict):
            candidates = result.get("results") or result.get("memories") or []
        else:
            candidates = result
        if not isinstance(candidates, list):
            return []
        return [item for item in candidates if isinstance(item, dict)]

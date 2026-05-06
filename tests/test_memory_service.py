"""Tests for Mem0-backed long-term memory integration."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.memory.service import MemoryService, format_memories_for_prompt


class TestMemoryService(unittest.TestCase):
    def test_format_memories_for_prompt_deduplicates_results(self):
        prompt = format_memories_for_prompt([
            {"memory": "User prefers Chinese."},
            {"memory": "User prefers Chinese."},
            {"text": "Workspace uses Python."},
            {"content": ""},
        ])

        self.assertIn("Relevant long-term memory:", prompt)
        self.assertEqual(prompt.count("User prefers Chinese."), 1)
        self.assertIn("- Workspace uses Python.", prompt)

    def test_builds_local_chroma_and_neo4j_mem0_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
                service = MemoryService(
                    {
                        "enabled": True,
                        "enable_graph": True,
                        "chroma": {
                            "collection_name": "clawd_code_memory_test",
                            "path": str(Path(temp_dir) / "chroma"),
                        },
                        "neo4j": {
                            "url": "bolt://localhost:7687",
                            "username": "neo4j",
                            "password": "password",
                            "database": "neo4j",
                            "threshold": 0.8,
                        },
                        "llm": {
                            "provider": "openai",
                            "model": "gpt-4.1-mini",
                            "api_key_env": "OPENAI_API_KEY",
                        },
                        "embedder": {
                            "provider": "openai",
                            "model": "text-embedding-3-small",
                            "api_key_env": "OPENAI_API_KEY",
                        },
                    },
                    workspace_root=Path(temp_dir),
                    session_id="session-test",
                )

                config = service._build_mem0_config()

        self.assertEqual(config["vector_store"]["provider"], "chroma")
        self.assertEqual(config["vector_store"]["config"]["collection_name"], "clawd_code_memory_test")
        self.assertIn("chroma", config["vector_store"]["config"]["path"])
        self.assertEqual(config["graph_store"]["provider"], "neo4j")
        self.assertEqual(config["graph_store"]["config"]["url"], "bolt://localhost:7687")
        self.assertEqual(config["llm"]["provider"], "openai")
        self.assertEqual(config["llm"]["config"]["api_key"], "sk-test")
        self.assertEqual(config["embedder"]["config"]["model"], "text-embedding-3-small")

    def test_disabled_service_does_not_initialize_client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = MemoryService({"enabled": False}, workspace_root=Path(temp_dir))

            self.assertEqual(service.search("hello"), [])
            service.add_turn("hello", "world")
            self.assertIsNone(service._client)

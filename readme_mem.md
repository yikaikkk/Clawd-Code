# Memory Service Configuration Guide

## Overview

The memory service provides long-term memory management using Mem0, supporting selective memory saving and automatic information extraction.

## Configuration Options

```json
{
  "memory": {
    "enabled": true,
    "debug": false,
    "inject": true,
    "write_after_turn": true,
    "search_limit": 5,
    "save_strategy": "all",
    "long_term_keywords": [],
    "enable_extraction": false,
    "extraction_prompt": "",
    "extraction_provider": {},
    "enable_graph": true,
    "user_id": "default",
    "agent_id": "clawd-code",
    "chroma": {
      "path": "~/.clawd/memory/chroma",
      "collection_name": "clawd_code_memory"
    },
    "neo4j": {
      "url": "",
      "username": "",
      "password": "",
      "database": "",
      "threshold": 0.7
    }
  }
}
```

## Save Strategies

| Strategy | Description |
|----------|-------------|
| `"all"` | Save all conversations (default) |
| `"long_term_only"` | Only save content containing user profile or long-term goals |

### Example: Save Only Long-term Memories

```json
{
  "memory": {
    "enabled": true,
    "save_strategy": "long_term_only"
  }
}
```

## Keyword-based Filtering

When `save_strategy` is set to `"long_term_only"`, the service filters memories based on keywords:

### Default Keywords

**User Profile:**
- Identity: 我是, 我的名字, 我叫, 我来自, 我从事, 我在
- Career: 我的职业, 我的工作, 我的专业, 我的学历, 我的背景
- Skills: 我擅长, 我熟悉, 我精通, 我了解, 我学习过
- Goals: 我的目标, 我的计划, 我想, 我希望, 我打算
- Preferences: 我的需求, 我的要求, 我需要, 我想要, 我的偏好

**Long-term Goals:**
- Planning: 长期, 计划, 规划, 目标, 愿景, 战略, 方向
- Time: 未来, 以后, 将来, 接下来, 准备, 打算
- Tasks: 项目, 任务, 里程碑, 阶段, 步骤

### Custom Keywords

```json
{
  "memory": {
    "enabled": true,
    "save_strategy": "long_term_only",
    "long_term_keywords": ["自定义关键词1", "自定义关键词2"]
  }
}
```

## Information Extraction

Automatically extract and summarize core information using LLM before saving.

### Configuration

```json
{
  "memory": {
    "enabled": true,
    "save_strategy": "long_term_only",
    "enable_extraction": true,
    "extraction_provider": {
      "provider": "qwen",
      "api_key_env": "QWEN_API_KEY",
      "model": "qwen3.5-flash",
      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
  }
}
```

### Custom Extraction Prompt

```json
{
  "memory": {
    "enable_extraction": true,
    "extraction_prompt": "请从以下对话中提取用户画像信息：\n\n用户输入: {user_input}\n助手回复: {assistant_output}\n\n输出格式：\n- 身份：\n- 目标：\n- 偏好："
  }
}
```

### Default Extraction Prompt

```
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
```

## Complete Example Configuration

```json
{
  "memory": {
    "enabled": true,
    "debug": true,
    "inject": true,
    "write_after_turn": true,
    "search_limit": 5,
    "save_strategy": "long_term_only",
    "enable_extraction": true,
    "extraction_provider": {
      "provider": "qwen",
      "api_key_env": "QWEN_API_KEY",
      "model": "qwen3.5-flash"
    },
    "chroma": {
      "path": "~/.clawd/memory/chroma"
    }
  }
}
```

## Workflow

1. **Receive Message**: User input and assistant response
2. **Check Strategy**: If `long_term_only`, filter by keywords
3. **Extract Info**: If `enable_extraction` is true, summarize using LLM
4. **Save Memory**: Store processed content to Mem0
5. **Inject Memory**: When responding, retrieve relevant memories

## Debug Mode

Enable debug logging to see memory operations:

```json
{
  "memory": {
    "enabled": true,
    "debug": true
  }
}
```

Debug output example:
```
[clawd:memory] initialized enabled=True inject=True write_after_turn=True
[clawd:memory] search started query_chars=42 limit=5 user_id=default
[clawd:memory] search succeeded memories=3
[clawd:memory] memory qualifies for long-term reason=user input contains long-term keywords
[clawd:memory] extraction started prompt_chars=200
[clawd:memory] extraction succeeded extracted_chars=150
[clawd:memory] write started user_chars=150 assistant_chars=150
[clawd:memory] write succeeded
```

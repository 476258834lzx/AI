"""LLM模型配置"""
import os
from enum import Enum


class LLMProvider(Enum):
    """LLM提供者枚举"""
    VLLM = "vllm"
    OLLAMA = "ollama"
    SGLANG = "sglang"


LLM_CONFIG = {
    "provider": LLMProvider.VLLM,
    "model": "/data/Workspace/models/Qwen/Qwen3-8B",
    "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    "embedding_model": "bge-m3",
    "embedding_url": os.getenv("EMBEDDING_URL", "http://localhost:11434"),
    "temperature": 0.7,
}


def get_llm_config():
    provider = LLM_CONFIG["provider"]

    if provider == LLMProvider.VLLM:
        return {
            "model": LLM_CONFIG["model"],
            "base_url": LLM_CONFIG["base_url"],
            "temperature": LLM_CONFIG["temperature"],
            "streaming": False,
            "provider": "vllm",
        }
    elif provider == LLMProvider.OLLAMA:
        return {
            "model": "qwen3.5",
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "temperature": LLM_CONFIG["temperature"],
            "streaming": False,
            "provider": "ollama",
        }
    elif provider == LLMProvider.SGLANG:
        return {
            "model": LLM_CONFIG["model"],
            "base_url": os.getenv("SGLANG_BASE_URL", "http://localhost:30000/v1"),
            "temperature": LLM_CONFIG["temperature"],
            "streaming": False,
            "provider": "sglang",
        }
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_embedding_config():
    return {
        "model": LLM_CONFIG["embedding_model"],
        "base_url": LLM_CONFIG["embedding_url"],
    }


def set_provider(provider: LLMProvider):
    """设置LLM提供者"""
    LLM_CONFIG["provider"] = provider
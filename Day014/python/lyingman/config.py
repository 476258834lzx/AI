"""LLM模型配置"""
import os

LLM_CONFIG = {
    "provider": "ollama",
    "model": "qwen3.5",
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "embedding_model": "bge-m3",
    "embedding_url": os.getenv("EMBEDDING_URL", "http://localhost:11434"),
    "temperature": 0.7,
}


def get_llm_config():
    return {
        "model": LLM_CONFIG["model"],
        "base_url": LLM_CONFIG["base_url"],
        "temperature": LLM_CONFIG["temperature"],
        "streaming": False,
    }


def get_embedding_config():
    return {
        "model": LLM_CONFIG["embedding_model"],
        "base_url": LLM_CONFIG["embedding_url"],
    }
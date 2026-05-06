#!/usr/bin/env python3
"""
Ollama 兼容的推理服务
使用 transformers 库模拟 Ollama 的 API
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, '/data/Workspace/airelearn/Day011/python/upload_huggingface')
sys.path.insert(0, '/data/Workspace/airelearn/Day011/python/my_llm/cache')

from model import StorierModel, StorierConfig
from tokenizer import SentencePieceTokenizer
from transformers import GenerationConfig
import json
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional


# 加载模型和 tokenizer
MODEL_PATH = "/data/Workspace/airelearn/Day011/python/my_llm/cache"

print("加载模型...")
tokenizer = SentencePieceTokenizer(
    model_file='tokenizer.model',
    vocab_file='tokenizer.vocab'
)
config = StorierConfig.from_pretrained(MODEL_PATH)
model = StorierModel.from_pretrained(MODEL_PATH, config=config)
model.eval()

print(f"模型加载完成: {type(model).__name__}")
print(f"Vocab size: {tokenizer.vocab_size}")


# FastAPI 应用
app = FastAPI(title="Storier Ollama-Compatible API")


class GenerateRequest(BaseModel):
    prompt: str
    options: Optional[dict] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = False


class ChatRequest(BaseModel):
    messages: List[dict]
    options: Optional[dict] = None
    stream: bool = False


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """生成文本"""
    # 解析参数
    max_tokens = req.options.get("num_predict", 100) if req.options else 100
    temperature = req.options.get("temperature", 0.7) if req.options else 0.7
    top_p = req.options.get("top_p", 0.9) if req.options else 0.9
    stop = req.options.get("stop", "<s>") if req.options else "<s>"

    # 准备输入
    prompt = req.prompt
    if req.template:
        prompt = req.template.replace("{{ .Prompt }}", prompt)

    inputs = tokenizer(prompt, return_tensors="pt")

    # 生成
    generation_config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            generation_config=generation_config
        )

    # 解码
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 移除输入部分
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]

    # 处理 stop token
    if stop in generated_text:
        generated_text = generated_text.split(stop)[0]

    return {
        "model": "storier",
        "response": generated_text,
        "done": True,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """聊天接口"""
    # 提取最后一条用户消息
    user_message = ""
    for msg in reversed(req.messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    # 准备提示
    prompt = f"<s>user\n{user_message}</s>\n<s>assistant\n"

    # 调用生成
    gen_req = GenerateRequest(
        prompt=prompt,
        options=req.options,
    )
    result = await generate(gen_req)

    return {
        "model": "storier",
        "message": {
            "role": "assistant",
            "content": result["response"],
        },
        "done": True,
    }


@app.get("/api/tags")
async def tags():
    """返回模型信息"""
    return {
        "models": [
            {
                "name": "storier",
                "model": "storier",
                "size": sum(p.numel() * p.element_size() for p in model.parameters()),
                "digest": "local",
                "details": {
                    "parent_model": "",
                    "format": "safetensors",
                    "family": "storier",
                    "families": ["storier"],
                    "parameter_size": f"{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B",
                    "quantization_level": "F32",
                }
            }
        ]
    }


@app.get("/")
async def root():
    return {"status": "ok", "model": "storier"}


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Storier Ollama-Compatible API Server")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Endpoints:")
    print(f"  - POST /api/generate  - 生成文本")
    print(f"  - POST /api/chat      - 聊天")
    print(f"  - GET  /api/tags      - 模型信息")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=11434)

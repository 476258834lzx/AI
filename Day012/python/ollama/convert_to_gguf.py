#!/usr/bin/env python3
"""
将 HuggingFace safetensors 模型转换为 GGUF 格式
用于 Ollama 部署
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open

from gguf import GGUFWriter


# 权重名称映射：原始名称 -> GGUF 名称
def convert_weight_name(name: str) -> str:
    """将原始权重名称转换为 GGUF 格式"""
    result = name

    # 处理 layer 索引 - 替换 tf_layer._layers.X 为 blk.X
    import re
    result = re.sub(r'tf_layer\._layers\.(\d+)', r'blk.\1', result)

    # 处理 attention 层
    result = result.replace("._att_layer._qw", ".attn_q")
    result = result.replace("._att_layer._kw", ".attn_k")
    result = result.replace("._att_layer._vw", ".attn_v")
    result = result.replace("._att_layer._ow", ".attn_output")

    # 处理 norm 层
    result = result.replace("._att_norm._w", ".attn_norm.weight")
    result = result.replace("._ffn_norm._w", ".ffn_norm.weight")

    # 处理 FFN 层
    result = result.replace("._ffn_layer._w0", ".ffn_gate")
    result = result.replace("._ffn_layer._w1", ".ffn_up")
    result = result.replace("._ffn_layer._w2", ".ffn_down")

    # 处理输出
    result = result.replace("tf_layer._out_norm._w", "output_norm.weight")
    result = result.replace("emb.weight", "token_embd.weight")

    # 移除前缀下划线
    result = result.lstrip("_")

    return result


def convert_to_gguf():
    """将模型转换为 GGUF 格式"""
    model_path = Path("/data/Workspace/airelearn/Day011/python/my_llm/cache")
    output_path = model_path / "model.gguf"

    print("=" * 60)
    print("转换模型到 GGUF 格式")
    print("=" * 60)

    # 加载配置
    config_path = model_path / "config.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # 获取模型参数
    vocab_size = config_dict.get("vocab_size", 50000)
    hidden_size = config_dict.get("input_dim", 768)
    num_layers = config_dict.get("num_layers", 48)
    num_heads = config_dict.get("n_q_heads", 12)
    num_kv_heads = config_dict.get("n_kv_heads", 2)
    max_pos = config_dict.get("max_pos_len", 16384)
    intermediate_size = config_dict.get("hide_dim", 3072)

    print(f"Vocab size: {vocab_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Num layers: {num_layers}")
    print(f"Num heads: {num_heads}")
    print(f"Num KV heads: {num_kv_heads}")
    print(f"Max position: {max_pos}")
    print(f"Intermediate size: {intermediate_size}")

    # 加载 tokenizer vocab
    vocab_file = model_path / "tokenizer.vocab"
    vocab_tokens = []
    token_scores = []

    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                token = parts[0]
                vocab_tokens.append(token)
                try:
                    score = float(parts[1]) if len(parts) > 1 else 0.0
                except ValueError:
                    score = 0.0
                token_scores.append(score)

    # 确保 vocab 大小正确
    while len(vocab_tokens) < vocab_size:
        vocab_tokens.append(f"<extra_id_{len(vocab_tokens)}>")
        token_scores.append(0.0)

    vocab_tokens = vocab_tokens[:vocab_size]
    token_scores = token_scores[:vocab_size]

    print(f"Loaded vocab with {len(vocab_tokens)} tokens")

    # 加载模型权重
    print("\n加载模型权重...")
    model_file = model_path / "model.safetensors"
    weights = {}

    with safe_open(model_file, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            weights[key] = tensor

    print(f"Loaded {len(weights)} tensors")

    # 创建 GGUF 文件
    print(f"\n写入 GGUF 文件到: {output_path}")

    writer = GGUFWriter(str(output_path), "llama")

    # 写入元数据
    writer.add_file_type(0)  # F32
    writer.add_vocab_size(vocab_size)
    writer.add_context_length(max_pos)
    writer.add_embedding_length(hidden_size)
    writer.add_block_count(num_layers)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(50000.0)

    # 添加词汇表
    writer.add_token_list(vocab_tokens)
    writer.add_token_scores(token_scores)

    # 添加特殊 token 映射
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(3)
    writer.add_pad_token_id(0)
    writer.add_unk_token_id(1)

    writer.write_header()
    writer.write_kv()

    # 写入权重
    print("\n转换权重...")
    for orig_name, tensor in weights.items():
        gguf_name = convert_weight_name(orig_name)

        # 转换为 float32
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        # 转换为 numpy
        np_tensor = tensor.numpy()

        # 写入张量
        writer.add_tensor(gguf_name, np_tensor)
        print(f"  {orig_name} -> {gguf_name}: {tensor.shape}")

    writer.write_tensors()
    writer.close()

    print(f"\n✓ GGUF 文件已创建: {output_path}")
    print(f"  文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return True


if __name__ == "__main__":
    success = convert_to_gguf()
    sys.exit(0 if success else 1)

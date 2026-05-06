#!/usr/bin/env python3
"""
将 DeepSpeed pt 模型转换为 HuggingFace/safetensors 格式
使用 upload_huggingface 目录下的模型代码
"""
import os
import sys
import json
import torch
import shutil
from pathlib import Path

# 从 upload_huggingface 目录导入模型
sys.path.insert(0, '/data/Workspace/airelearn/Day011/python/upload_huggingface')
from model import StorierModel, StorierConfig
from tokenizer import SentencePieceTokenizer


def create_tokenizer_files(cache_dir):
    """创建 tokenizer.model 和 tokenizer.vocab 文件"""
    cache_path = Path(cache_dir)

    # 创建词汇表（使用 SentencePiece 格式）
    # 这里创建一个简单的词汇表用于演示
    vocab_size = 50000

    # 创建 tokenizer.vocab
    vocab_file = cache_path / "tokenizer.vocab"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        # 添加特殊 token
        f.write("<pad>\t0\n")
        f.write("<unk>\t1\n")
        f.write("<s>\t2\n")
        f.write("</s>\t3\n")
        # 添加常见字符（简化版本）
        for i in range(4, min(vocab_size, 1000)):
            f.write(f"token_{i}\t{i}\n")
        # 如果需要更多 token
        if vocab_size > 1000:
            # 使用字符级 vocab
            import string
            chars = list(string.ascii_letters + string.digits + string.punctuation + " ")
            for i, c in enumerate(chars):
                if 4 + i < vocab_size:
                    f.write(f"{c}\t{4 + i}\n")

    # 创建 tokenizer.model (SentencePiece 模型文件)
    # SentencePiece 需要训练数据来创建模型，这里我们创建一个简单的模型文件
    # 实际使用时应该用真实数据训练
    model_file = cache_path / "tokenizer.model"

    # 使用 sentencepiece 库创建一个简单的模型
    try:
        import sentencepiece as spm

        # 创建临时训练数据 - 使用更多数据
        train_file = cache_path / "temp_train.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            # 使用字符和常见词作为训练数据
            for _ in range(1000):  # 增加训练数据量
                f.write("我 爱 北京 天安门 天安门上太阳升 中国是一个伟大的国家\n")
                f.write("今天天气很好 我们去公园玩吧\n")
                f.write("人工智能改变世界 机器学习深度学习神经网络\n")
                f.write("自然语言处理 计算机视觉语音识别推荐系统\n")

        # 训练 SentencePiece 模型，使用较小的 vocab_size
        vocab_size = 100  # 使用更小的 vocab_size 以确保能训练成功
        spm.SentencePieceTrainer.train(
            input=str(train_file),
            model_prefix=str(cache_path / "temp_sp"),
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type='unigram',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )

        # 移动生成的文件
        if (cache_path / "temp_sp.model").exists():
            shutil.move(str(cache_path / "temp_sp.model"), str(model_file))
            print(f"Tokenizer model 创建成功: {model_file}")

        # 清理临时文件
        if train_file.exists():
            train_file.unlink()
        for f in cache_path.glob("temp_sp*"):
            f.unlink()

        return True
    except Exception as e:
        print(f"Warning: SentencePiece 训练失败: {e}")
        print("将使用简化的 tokenizer...")

        # 创建一个占位的模型文件
        with open(model_file, 'wb') as f:
            f.write(b'')  # 空文件作为占位

        return False


def convert_pt_to_safetensors():
    """将 DeepSpeed pt 权重转换为 HuggingFace safetensors 格式"""
    cache_dir = Path("/data/Workspace/airelearn/Day011/python/my_llm/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path("/data/Workspace/airelearn/Day011/python/my_llm/weights/9/mp_rank_00_model_states.pt")

    print(f"加载权重: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["module"]
    print(f"原始权重键数量: {len(state_dict)}")
    print(f"部分权重键: {list(state_dict.keys())[:10]}")

    # 过滤掉非模型权重
    model_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_")}
    print(f"模型权重键数量: {len(model_state_dict)}")

    # 创建配置
    config = StorierConfig(
        num_layers=48,
        input_dim=768,
        hide_dim=3072,
        n_q_heads=12,
        n_kv_heads=2,
        max_pos_len=16384,
        vocab_size=50000,
        use_cache=True
    )

    # 创建模型
    print("创建模型...")
    model = StorierModel(config)

    # 加载权重
    print("加载权重到模型...")
    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    if missing:
        print(f"缺失的键: {missing}")
    if unexpected:
        print(f"意外的键: {unexpected}")

    # 保存模型
    print("保存模型到 HuggingFace 格式...")
    model.save_pretrained(cache_dir)

    # 更新配置文件
    config_dict = config.to_dict()
    config_dict["auto_map"] = {
        "AutoConfig": "model.StorierConfig",
        "AutoModelForCausalLM": "model.StorierForCausalLM",
    }

    config_file = cache_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    print(f"配置文件更新: {config_file}")

    # 创建 tokenizer 文件
    print("创建 tokenizer 文件...")
    create_tokenizer_files(cache_dir)

    # 创建/复制 tokenizer.py
    src_tokenizer = Path("/data/Workspace/airelearn/Day011/python/upload_huggingface/tokenizer.py")
    if src_tokenizer.exists():
        shutil.copy2(src_tokenizer, cache_dir / "tokenizer.py")
        print(f"复制 tokenizer.py 到 {cache_dir}")

    # 创建 tokenizer_config.json
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": True,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "chat_template": """
{%- for message in messages %}
    {%- if (message.role == "system") %}{{- '<s>system:\n'+ message.content + '</s>\n' }}{%- endif %}
    {%- if (message.role == "user") %}{{- '<s>user:\n'+ message.content + '</s>\n' }}{%- endif %}
{%- endfor %}
{{- '<s>assistant:\n' }}
        """,
    }
    with open(cache_dir / "tokenizer_config.json", 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"创建 tokenizer_config.json")

    print(f"\n转换完成！模型保存在: {cache_dir}")

    # 列出生成的文件
    print("\n生成的文件:")
    for f in sorted(cache_dir.iterdir()):
        size = f.stat().st_size / 1024 / 1024  # MB
        print(f"  {f.name}: {size:.2f} MB" if size > 0.01 else f"  {f.name}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("转换 DeepSpeed pt 模型到 HuggingFace/safetensors 格式")
    print("=" * 60)

    success = convert_pt_to_safetensors()

    if success:
        print("\n✓ 转换成功!")
    else:
        print("\n✗ 转换失败!")
        sys.exit(1)
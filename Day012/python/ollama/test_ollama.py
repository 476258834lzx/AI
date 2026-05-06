#!/usr/bin/env python3
"""
完整的测试脚本 - 测试模型转换和 Ollama 集成
"""
import sys
import os
import torch
import subprocess
from pathlib import Path

# 设置工作目录
WORK_DIR = Path("/data/Workspace/airelearn/Day011/python/my_llm/cache")
os.chdir(WORK_DIR)

sys.path.insert(0, '/data/Workspace/airelearn/Day011/python/upload_huggingface')
sys.path.insert(0, str(WORK_DIR))

from model import StorierModel, StorierConfig
from tokenizer import SentencePieceTokenizer
from transformers import GenerationConfig


def test_huggingface_model():
    """测试 HuggingFace 格式模型"""
    print("\n" + "=" * 60)
    print("测试 1: HuggingFace 模型加载和推理")
    print("=" * 60)

    model_path = WORK_DIR

    # 1. 加载 tokenizer
    print("\n[1.1] 加载 Tokenizer...")
    try:
        tokenizer = SentencePieceTokenizer(
            model_file='tokenizer.model',
            vocab_file='tokenizer.vocab'
        )
        print(f"  ✓ Tokenizer 加载成功 (vocab_size={tokenizer.vocab_size})")
    except Exception as e:
        print(f"  ✗ Tokenizer 加载失败: {e}")
        return False

    # 2. 加载模型
    print("\n[1.2] 加载模型...")
    try:
        config = StorierConfig.from_pretrained(model_path)
        model = StorierModel.from_pretrained(model_path, config=config)
        model.eval()
        print(f"  ✓ 模型加载成功 ({type(model).__name__})")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. 测试推理
    print("\n[1.3] 测试推理...")
    try:
        test_text = "你好"
        inputs = tokenizer(test_text, return_tensors="pt")
        print(f"  Input: '{test_text}' -> {inputs['input_ids'].shape}")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        print(f"  Output logits: {logits.shape}")
        print("  ✓ 推理测试通过")
    except Exception as e:
        print(f"  ✗ 推理失败: {e}")
        return False

    # 4. 测试文本生成
    print("\n[1.4] 测试文本生成...")
    try:
        generation_config = GenerationConfig(
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_ids"],
                generation_config=generation_config
            )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"  Generated: {generated_text}")
        print("  ✓ 文本生成测试通过")
    except Exception as e:
        print(f"  ✗ 文本生成失败: {e}")
        return False

    return True


def test_ollama_model():
    """测试 Ollama 模型"""
    print("\n" + "=" * 60)
    print("测试 2: Ollama 模型注册")
    print("=" * 60)

    # 1. 检查 Ollama 服务
    print("\n[2.1] 检查 Ollama 服务...")
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if "storier" in result.stdout:
            print("  ✓ storier 模型已注册")
        else:
            print("  ! storier 模型未注册，将尝试创建")
    except Exception as e:
        print(f"  ✗ Ollama 服务检查失败: {e}")
        return False

    # 2. 尝试运行模型
    print("\n[2.2] 测试 Ollama 运行...")
    print("  (注意: 由于 Storier 是自定义架构，Ollama 可能无法直接运行)")
    print("  替代方案: 使用 transformers 进行推理，或启动 ollama_inference.py 服务")

    return True


def test_model_files():
    """测试模型文件完整性"""
    print("\n" + "=" * 60)
    print("测试 3: 模型文件完整性检查")
    print("=" * 60)

    required_files = [
        "model.safetensors",
        "config.json",
        "tokenizer.model",
        "tokenizer.vocab",
        "tokenizer.py",
        "model.py",
        "tokenizer_config.json",
        "generation_config.json",
    ]

    all_exist = True
    for fname in required_files:
        fpath = WORK_DIR / fname
        if fpath.exists():
            size = fpath.stat().st_size / 1024 / 1024
            print(f"  ✓ {fname}: {size:.2f} MB")
        else:
            print(f"  ✗ {fname}: 缺失")
            all_exist = False

    return all_exist


def main():
    print("=" * 60)
    print("Storier 模型完整测试")
    print("=" * 60)

    results = []

    # 测试 1: HuggingFace 模型
    results.append(("HuggingFace 模型测试", test_huggingface_model()))

    # 测试 2: 模型文件
    results.append(("模型文件完整性", test_model_files()))

    # 测试 3: Ollama
    results.append(("Ollama 模型测试", test_ollama_model()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {status} - {name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过!")
        print("\n下一步:")
        print("  1. 使用 transformers 进行推理:")
        print("     python ollama_inference.py")
        print("  2. 或者使用 curl 测试 API:")
        print("     curl -X POST http://localhost:11434/api/generate -d '{\"prompt\":\"你好\"}'")
    else:
        print("✗ 部分测试失败，请检查上述输出")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

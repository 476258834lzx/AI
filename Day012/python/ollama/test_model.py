#!/usr/bin/env python3
"""
测试转换后的模型是否正常工作
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, '/data/Workspace/airelearn/Day011/python/upload_huggingface')
sys.path.insert(0, '/data/Workspace/airelearn/Day011/python/my_llm/cache')

from model import StorierModel, StorierConfig
from tokenizer import SentencePieceTokenizer
from transformers import GenerationConfig


def test_model():
    """测试模型加载和推理"""
    model_path = "/data/Workspace/airelearn/Day011/python/my_llm/cache"

    print("=" * 60)
    print("测试模型加载")
    print("=" * 60)

    # 1. 测试加载 tokenizer
    print("\n1. 加载 Tokenizer...")
    try:
        tokenizer = SentencePieceTokenizer(
            model_file='tokenizer.model',
            vocab_file='tokenizer.vocab'
        )
        print(f"   ✓ Tokenizer 加载成功")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        print(f"   PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
        print(f"   BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
        print(f"   EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    except Exception as e:
        print(f"   ✗ Tokenizer 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. 测试分词
    print("\n2. 测试分词...")
    try:
        test_text = "你好世界"
        tokens = tokenizer.tokenize(test_text)
        print(f"   Input: {test_text}")
        print(f"   Tokens: {tokens}")
        token_ids = tokenizer.encode(test_text)
        print(f"   Token IDs: {token_ids}")
        decoded = tokenizer.decode(token_ids)
        print(f"   Decoded: {decoded}")
        print("   ✓ 分词测试通过")
    except Exception as e:
        print(f"   ✗ 分词测试失败: {e}")
        return False

    # 3. 测试加载模型
    print("\n3. 加载模型...")
    try:
        config = StorierConfig.from_pretrained(model_path)
        model = StorierModel.from_pretrained(model_path, config=config)
        model.eval()
        print(f"   ✓ 模型加载成功")
        print(f"   Model type: {type(model).__name__}")
    except Exception as e:
        print(f"   ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 测试模型推理
    print("\n4. 测试模型推理...")
    try:
        # 使用 CPU 进行推理
        model = model.cpu()

        # 准备输入
        input_text = "你好"
        inputs = tokenizer(input_text, return_tensors="pt")
        print(f"   Input: {input_text}")
        print(f"   Input IDs shape: {inputs['input_ids'].shape}")

        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        print(f"   Output logits shape: {logits.shape}")
        print(f"   ✓ 前向传播测试通过")

        # 5. 测试生成（简单生成）
        print("\n5. 测试文本生成...")
        generation_config = GenerationConfig(
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or 0,
        )

        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_ids"],
                generation_config=generation_config
            )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"   Generated: {generated_text}")
        print(f"   ✓ 文本生成测试通过")

    except Exception as e:
        print(f"   ✗ 生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("所有测试通过！模型可以正常工作。")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)

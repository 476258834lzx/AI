#!/usr/bin/env python3
"""
Storier 命令行推理脚本
使用 transformers 库进行本地推理
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, '/data/Workspace/airelearn/Day011/python/upload_huggingface')
sys.path.insert(0, '/data/Workspace/airelearn/Day011/python/my_llm/cache')

from model import StorierModel, StorierConfig
from tokenizer import SentencePieceTokenizer
from transformers import GenerationConfig


MODEL_PATH = "/data/Workspace/airelearn/Day011/python/my_llm/cache"

# ---------- 加载模型和 tokenizer ----------
print("加载模型...")
tokenizer = SentencePieceTokenizer(
    model_file='tokenizer.model',
    vocab_file='tokenizer.vocab'
)
config = StorierConfig.from_pretrained(MODEL_PATH)
model = StorierModel.from_pretrained(MODEL_PATH, config=config)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"推理设备: {device}")

print(f"模型加载完成: {type(model).__name__}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print("=" * 60)


def generate_text(prompt: str,
                  max_new_tokens: int = 100,
                  temperature: float = 0.7,
                  top_p: float = 0.9,
                  stop: str = "<s>"):
    """执行推理并返回生成的文本"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            generation_config=generation_config,
        )

    # 解码
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 移除输入部分
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]

    # 处理 stop token
    if stop in generated_text:
        generated_text = generated_text.split(stop)[0]

    return generated_text.strip()


def interactive_mode(**gen_kwargs):
    """交互式对话模式"""
    print("\n进入交互模式 (输入 'quit' / 'exit' 退出)")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if not user_input:
            continue

        prompt = f"<s>user\n{user_input}</s>\n<s>assistant\n"

        print("助手: ", end="", flush=True)
        response = generate_text(prompt, **gen_kwargs)
        print(response)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Storier 命令行推理工具")
    parser.add_argument("prompt",default="凌风渊", nargs="?", help="待推理的提示文本（不指定则进入交互模式）")
    parser.add_argument("--max-tokens", type=int, default=100, help="最大生成 token 数 (默认 100)")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度 (默认 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P (默认 0.9)")
    parser.add_argument("--stop", type=str, default="<s>", help="停止 token")
    parser.add_argument("--interactive", "-i", action="store_true", help="强制进入交互模式")

    args = parser.parse_args()

    gen_kwargs = dict(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop,
    )

    if args.interactive or args.prompt is None:
        # 交互模式
        interactive_mode(**gen_kwargs)
    else:
        # 单次推理
        prompt = f"<s>user\n{args.prompt}</s>\n<s>assistant\n"
        print("助手: ", end="", flush=True)
        response = generate_text(prompt, **gen_kwargs)
        print(response)

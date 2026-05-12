"""
主训练脚本

运行最小模型训练并测试
"""

import os
import sys
import torch
import random
import argparse
from datetime import datetime

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(__file__))

from config import ModelConfig, TrainingConfig, ConfigFactory
from modeling_llama import LlamaForCausalLM
from dataset import RandomTextDataset, save_synthetic_images
from trainer import Trainer


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_mini_model(
    output_dir: str = "./output",
    num_epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-3,
    seq_length: int = 32,
    num_samples: int = 100,
    device: str = None
):
    """
    训练最小模型

    参数:
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        seq_length: 序列长度
        num_samples: 训练样本数
        device: 设备
    """
    print("="*60)
    print("Training Mini Llama Model")
    print("="*60)

    # 设置设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 设置随机种子
    set_seed(42)

    # 创建模型配置
    config = ConfigFactory.create_mini_model()
    print(f"\nModel Config:")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Intermediate size: {config.intermediate_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Num heads: {config.num_attention_heads}")
    print(f"  - Num KV heads: {config.num_key_value_heads}")
    print(f"  - Head dim: {config.head_dim}")
    print(f"  - Max position: {config.max_position_embeddings}")

    # 创建模型
    print("\nCreating model...")
    model = LlamaForCausalLM(config)
    model.to(device)

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable_params:,}")

    # 创建数据集
    print(f"\nCreating dataset with {num_samples} samples...")
    train_dataset = RandomTextDataset(
        num_samples=num_samples,
        seq_length=seq_length,
        vocab_size=config.vocab_size
    )
    print(f"  Dataset size: {len(train_dataset)}")

    # 创建数据加载器
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # 创建训练配置
    training_config = TrainingConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=5,
        save_steps=50,
        fp16=torch.cuda.is_available()
    )

    # 创建训练器
    print("\nCreating trainer...")
    trainer = Trainer(
        model=model,
        config=training_config,
        device=device
    )

    # 开始训练
    print("\nStarting training...")
    trainer.train(train_loader)

    # 保存最终模型
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    print(f"\nSaving model to {final_dir}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict()
    }, os.path.join(final_dir, "model.pt"))

    print("\nTraining completed!")

    # 测试生成
    print("\n" + "="*60)
    print("Testing Generation")
    print("="*60)

    model.eval()
    test_input = torch.randint(4, config.vocab_size, (1, 8), device=device)

    print(f"Input tokens: {test_input.tolist()}")

    with torch.no_grad():
        output = model.generate(
            test_input,
            max_new_tokens=16,
            temperature=0.8,
            top_k=20
        )

    print(f"Output tokens: {output.tolist()}")
    print(f"Output shape: {output.shape}")

    return model, config


def test_output_format(model, config, device='cpu'):
    """测试输出格式"""
    print("\n" + "="*60)
    print("Testing Output Format")
    print("="*60)

    model.eval()

    # 创建测试输入
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(4, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(4, config.vocab_size, (batch_size, seq_len), device=device)

    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    # 检查输出格式
    print("\nOutput Keys:", list(outputs.keys()))

    print(f"\nLogits:")
    print(f"  - Shape: {outputs['logits'].shape}")
    print(f"  - Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
    print(f"  - Dtype: {outputs['logits'].dtype}")

    print(f"\nLoss:")
    if outputs['loss'] is not None:
        print(f"  - Value: {outputs['loss'].item():.4f}")
        print(f"  - Shape: {outputs['loss'].shape}")
    else:
        print("  - None")

    # 验证格式
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size), \
        "Logits shape mismatch!"

    print("\nOutput format test passed!")


def generate_and_save_images(output_dir: str, num_images: int = 10):
    """生成并保存合成图像"""
    print("\n" + "="*60)
    print(f"Generating {num_images} Synthetic Images")
    print("="*60)

    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    image_paths = save_synthetic_images(
        output_dir=image_dir,
        num_images=num_images,
        image_size=(224, 224)
    )

    print(f"\nGenerated {len(image_paths)} images:")
    for path in image_paths[:5]:
        print(f"  - {path}")
    if len(image_paths) > 5:
        print(f"  ... and {len(image_paths) - 5} more")

    return image_paths


def main():
    parser = argparse.ArgumentParser(description="Train mini Llama model")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--seq_length", type=int, default=32,
                       help="Sequence length")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of training samples")
    parser.add_argument("--num_images", type=int, default=10,
                       help="Number of synthetic images to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 生成合成图像
    generate_and_save_images(args.output_dir, args.num_images)

    # 训练模型
    model, config = train_mini_model(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_length=args.seq_length,
        num_samples=args.num_samples,
        device=device
    )

    # 测试输出格式
    test_output_format(model, config, device)

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

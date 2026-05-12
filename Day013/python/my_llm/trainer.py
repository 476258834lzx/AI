"""
训练脚本

支持纯文本训练和多模态训练
包含 DeepSpeed 支持（可选）
"""

import os
import sys
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

# 本地模块
from config import ModelConfig, TrainingConfig, ConfigFactory
from modeling_llama import LlamaForCausalLM
from modeling_vlm import LlamaForConditionalGeneration
from dataset import DatasetFactory, RandomTextDataset, MultimodalDataset


# =============================================================================
# 训练器
# =============================================================================
class Trainer:
    """
    模型训练器

    支持:
    - 纯文本训练
    - 多模态训练
    - 梯度累积
    - 混合精度训练
    - DeepSpeed（可选）
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 移到设备
        self.model.to(self.device)

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 混合精度
        self.scaler = GradScaler() if config.fp16 else None

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # 日志
        self.logger = Logger(config.output_dir)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        # 分层学习率
        no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        if self.config.optimizer.lower() == 'adamw':
            return AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
        elif self.config.optimizer.lower() == 'sgd':
            return torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self):
        """创建学习率调度器"""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_train_epochs,
            eta_min=self.config.learning_rate * 0.1
        )

    def train(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None):
        """
        训练循环

        参数:
            train_loader: 训练数据加载器
            eval_loader: 验证数据加载器（可选）
        """
        self.model.train()

        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(train_loader):
                # 移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # 前向传播
                with autocast(enabled=self.config.fp16):
                    outputs = self.model(**batch)
                    loss = outputs.get('loss', outputs.get('logits', None))

                    if loss is None:
                        # 对于没有 labels 的情况，计算 logits 损失
                        logits = outputs['logits']
                        labels = batch.get('labels', batch.get('input_ids'))
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
                        loss = loss_fct(
                            shift_logits.view(-1, logits.size(-1)),
                            shift_labels.view(-1)
                        )

                # 梯度累积
                loss = loss / self.config.gradient_accumulation_steps

                # 缩放损失（用于混合精度）
                if self.scaler:
                    scaled_loss = self.scaler.scale(loss)
                else:
                    scaled_loss = loss

                scaled_loss.backward()

                # 梯度累积步数
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # 取消缩放并裁剪梯度
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # 更新权重
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.global_step += 1

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1

                # 日志
                if step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logger.log({
                        'step': self.global_step,
                        'epoch': epoch,
                        'loss': avg_loss,
                        'lr': lr
                    })
                    print(f"Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

            # Epoch 结束
            avg_epoch_loss = epoch_loss / num_batches
            self.logger.log({
                'epoch': epoch,
                'avg_loss': avg_epoch_loss,
                'type': 'epoch'
            })
            print(f"Epoch {epoch} completed | Avg Loss: {avg_epoch_loss:.4f}")

            # 评估
            if eval_loader:
                eval_loss = self.evaluate(eval_loader)
                print(f"Eval Loss: {eval_loss:.4f}")

            # 保存检查点
            self.save_checkpoint(f"checkpoint-epoch-{epoch}")

            # 更新学习率
            self.scheduler.step()

        # 保存最终模型
        self.save_checkpoint("final")

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in eval_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs.get('loss', None)

            if loss is None:
                logits = outputs['logits']
                labels = batch.get('labels', batch.get('input_ids'))
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)
                loss = loss_fct(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1)
                )

            total_loss += loss.item()
            num_batches += 1

        self.model.train()
        return total_loss / num_batches

    def save_checkpoint(self, name: str):
        """保存检查点"""
        output_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(output_dir, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, os.path.join(output_dir, "checkpoint.pt"))
        print(f"Checkpoint saved to {output_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}")


# =============================================================================
# 日志记录器
# =============================================================================
class Logger:
    """简单日志记录器"""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "train.log")
        os.makedirs(log_dir, exist_ok=True)

    def log(self, data: Dict[str, Any]):
        """记录日志"""
        timestamp = datetime.now().isoformat()
        log_line = f"[{timestamp}] {data}\n"

        with open(self.log_file, 'a') as f:
            f.write(log_line)


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train Llama model")
    parser.add_argument("--model_size", type=str, default="mini",
                       choices=["mini", "small", "vlm"],
                       help="Model size: mini, small, or vlm")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--seq_length", type=int, default=64,
                       help="Sequence length")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of training samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda or cpu)")

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建设备
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建配置
    model_config = ConfigFactory.create_mini_model()
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=50
    )

    # 创建模型
    print(f"Creating {args.model_size} model...")
    model = LlamaForCausalLM(model_config)

    # 创建数据集
    print(f"Creating dataset with {args.num_samples} samples...")
    train_dataset = RandomTextDataset(
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        vocab_size=model_config.vocab_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # 创建训练器
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        config=training_config,
        device=device
    )

    # 开始训练
    print("Starting training...")
    trainer.train(train_loader)

    print("Training completed!")


if __name__ == "__main__":
    main()

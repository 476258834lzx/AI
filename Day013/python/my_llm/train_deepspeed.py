"""
DeepSpeed 训练器

参考 Day011 pretrain.py 的 DeepSpeed 实现方式
适配 Day013 的 LlamaForCausalLM 模型

用法:
    deepspeed train_deepspeed.py --deepspeed_config deepspeed_config.json
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import deepspeed

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from modeling_llama import LlamaForCausalLM
from dataset import RandomTextDataset
from config import ConfigFactory


def parse_arguments():
    parser = argparse.ArgumentParser(description='DeepSpeed Llama Training')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--ss', default=0, type=int)
    parser.add_argument('--paragraphsize', default=64, type=int)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--max_steps', default=None, type=int, help='最大训练步数，None则遍历整个数据集')

    # deepspeed.add_config_arguments 会添加 --deepspeed_config 参数
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # 设置默认的 deepspeed_config 路径
    if args.deepspeed_config is None:
        args.deepspeed_config = 'deepspeed_config.json'

    return args

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


class Trainer:
    def __init__(self):
        deepspeed.init_distributed()

        self.args = parse_arguments()
        rank = deepspeed.comm.get_rank()

        # 日志（仅主进程）
        if rank == 0 and HAS_TENSORBOARD:
            self.log = SummaryWriter("runs")
        else:
            self.log = None

        # 创建模型
        model_config = ConfigFactory.create_mini_model()
        self.model = LlamaForCausalLM(model_config)
        self.vocab_size = model_config.vocab_size

        # 创建数据集
        dataset = RandomTextDataset(
            num_samples=self.args.num_samples,
            seq_length=self.args.paragraphsize,
            vocab_size=self.vocab_size
        )

        # 初始化 DeepSpeed（与 Day011 pretrain.py 一致）
        # config 通过 args.deepspeed_config 传递
        from torch.optim import AdamW
        optimizer = AdamW(self.model.parameters(), lr=1e-4)

        self.engine, self.opt, self.training_dataloader, self.lr_scheduler = deepspeed.initialize(
            args=self.args,
            model=self.model,
            optimizer=optimizer,
            training_data=dataset,
            model_parameters=self.model.parameters(),
        )

        # 损失函数忽略 pad token (id=0)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def __call__(self):
        rank = deepspeed.comm.get_rank()
        self.engine.train()

        # 加载检查点（如果存在）
        _, client_sd = self.engine.load_checkpoint("weights")
        if client_sd is None or "step" not in client_sd:
            client_sd = {"step": 0}

        max_steps = self.args.max_steps

        for i, batch in enumerate(self.training_dataloader):
            # 处理 Day013 RandomTextDataset 返回的 dict
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
            else:
                input_ids = batch

            ds = input_ids.to(device=self.engine.device, dtype=torch.long)

            # 因果推理：xs 是输入，ys 是目标
            xs = ds[:, :-1]
            ys = ds[:, 1:]

            # 前向传播
            outputs = self.engine(xs)

            # 提取 logits（LlamaForCausalLM 返回 dict）
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            # 转换形状 [B, seq_len, vocab_size] -> [seq_len * B, vocab_size]
            logits = logits.reshape(-1, self.vocab_size)
            ys = ys.reshape(-1)

            # 计算损失
            loss = self.loss_fn(logits, ys)

            # DeepSpeed 反向传播和优化
            self.engine.backward(loss)
            self.engine.step()

            step = client_sd["step"]
            if rank == 0:
                if i % 10 == 0:
                    print(f"Step {step} | Loss: {loss.item():.4f}")
                    if self.log and HAS_TENSORBOARD:
                        self.log.add_scalar("loss", loss.item(), step)

            client_sd["step"] += 1

            # 限制最大步数
            if max_steps is not None and i >= max_steps - 1:
                break

        # 保存检查点
        save_tag = self.args.ss
        self.engine.save_checkpoint("weights", tag=f"{save_tag}", client_state={"step": client_sd['step']})


if __name__ == '__main__':
    train = Trainer()
    train()

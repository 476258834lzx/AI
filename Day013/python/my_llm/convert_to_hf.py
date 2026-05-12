"""
模型转换脚本

将自定义模型格式转换为 HuggingFace 格式

支持:
1. 从 PyTorch checkpoint 转换到 HuggingFace safetensors 格式
2. 生成 config.json 和 tokenizer 文件
3. 适配标准 Llama 格式
"""

import os
import json
import torch
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from config import ModelConfig, VisionConfig
from modeling_llama import LlamaForCausalLM, LlamaModel


# =============================================================================
# 权重名称映射
# =============================================================================
# 自定义模型权重名称 -> HuggingFace 标准名称
WEIGHT_MAPPING = {
    # Embedding
    'model.embed_tokens.weight': 'model.embed_tokens.weight',

    # Decoder Layers
    'model.layers.{i}.input_layernorm.weight': 'model.layers.{i}.input_layernorm.weight',
    'model.layers.{i}.self_attn.q_proj.weight': 'model.layers.{i}.self_attn.q_proj.weight',
    'model.layers.{i}.self_attn.k_proj.weight': 'model.layers.{i}.self_attn.k_proj.weight',
    'model.layers.{i}.self_attn.v_proj.weight': 'model.layers.{i}.self_attn.v_proj.weight',
    'model.layers.{i}.self_attn.o_proj.weight': 'model.layers.{i}.self_attn.o_proj.weight',
    'model.layers.{i}.post_attention_layernorm.weight': 'model.layers.{i}.post_attention_layernorm.weight',
    'model.layers.{i}.mlp.gate_proj.weight': 'model.layers.{i}.mlp.gate_proj.weight',
    'model.layers.{i}.mlp.up_proj.weight': 'model.layers.{i}.mlp.up_proj.weight',
    'model.layers.{i}.mlp.down_proj.weight': 'model.layers.{i}.mlp.down_proj.weight',

    # Final LayerNorm
    'model.norm.weight': 'model.norm.weight',

    # LM Head
    'lm_head.weight': 'lm_head.weight',
}


# =============================================================================
# HuggingFace 配置模板
# =============================================================================
HF_CONFIG_TEMPLATE = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 2,
    "eos_token_id": 3,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "max_position_embeddings": 512,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "4.35.0",
    "use_cache": True,
    "vocab_size": 4096
}


# =============================================================================
# 转换器
# =============================================================================
class ModelConverter:
    """
    模型格式转换器

    将自定义模型转换为 HuggingFace 格式
    """

    def __init__(
        self,
        config: ModelConfig,
        save_dir: str
    ):
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def convert_model(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        转换模型权重

        参数:
            state_dict: 原始状态字典

        返回:
            转换后的状态字典
        """
        converted = {}

        for key, value in state_dict.items():
            # 跳过非权重参数
            if not isinstance(value, torch.Tensor):
                continue

            # 名称映射
            new_key = self._map_weight_name(key)
            if new_key:
                converted[new_key] = value
            else:
                # 保留未映射的权重
                converted[key] = value

        return converted

    def _map_weight_name(self, name: str) -> Optional[str]:
        """映射权重名称"""
        # 处理层的索引
        for i in range(32):  # 假设最多 32 层
            for pattern in WEIGHT_MAPPING:
                if pattern.format(i=i) == name:
                    return WEIGHT_MAPPING[pattern].format(i=i)

        return name

    def save_config(self, output_dir: Optional[str] = None):
        """保存配置文件"""
        if output_dir is None:
            output_dir = self.save_dir

        config_dict = HF_CONFIG_TEMPLATE.copy()

        # 更新为实际配置
        config_dict.update({
            'vocab_size': self.config.vocab_size,
            'hidden_size': self.config.hidden_size,
            'intermediate_size': self.config.intermediate_size,
            'num_hidden_layers': self.config.num_hidden_layers,
            'num_attention_heads': self.config.num_attention_heads,
            'num_key_value_heads': self.config.num_key_value_heads,
            'head_dim': self.config.head_dim,
            'max_position_embeddings': self.config.max_position_embeddings,
            'rope_theta': self.config.rope_theta,
            'rms_norm_eps': self.config.rms_norm_eps,
            'initializer_range': self.config.initializer_range,
            'pad_token_id': self.config.pad_token_id,
            'bos_token_id': self.config.bos_token_id,
            'eos_token_id': self.config.eos_token_id,
        })

        # 如果有视觉配置
        if self.config.vision_config:
            config_dict['model_type'] = 'llava'
            config_dict['vision_config'] = self.config.vision_config.to_dict()

        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Config saved to {config_path}")
        return config_path

    def save_model(self, state_dict: Dict[str, torch.Tensor], output_dir: Optional[str] = None):
        """保存模型权重"""
        if output_dir is None:
            output_dir = self.save_dir

        # 转换权重
        converted = self.convert_model(state_dict)

        # 保存为 safetensors
        try:
            from safetensors.torch import save_file

            # 分割成多个文件（Llama 格式）
            shard_size = 10 * 1024 * 1024 * 1024  # 10GB per shard
            total_size = sum(v.nelement() * v.element_size() for v in converted.values())

            if total_size < shard_size:
                # 小模型，直接保存
                output_path = os.path.join(output_dir, 'model.safetensors')
                save_file(converted, output_path)
                print(f"Model saved to {output_path}")
            else:
                # 大模型，分片保存
                self._save_sharded(converted, output_dir)

        except ImportError:
            # 如果没有 safetensors，使用 torch
            output_path = os.path.join(output_dir, 'pytorch_model.bin')
            torch.save(converted, output_path)
            print(f"Model saved to {output_path}")

        return output_dir

    def _save_sharded(self, state_dict: Dict[str, torch.Tensor], output_dir: str):
        """保存分片模型"""
        from safetensors.torch import save_file

        shard_size = 10 * 1024 * 1024 * 1024  # 10GB
        current_shard = {}
        current_size = 0
        shard_index = 0

        for key, value in state_dict.items():
            weight_size = value.nelement() * value.element_size()

            if current_size + weight_size > shard_size and current_shard:
                # 保存当前分片
                shard_path = os.path.join(output_dir, f'model-{shard_index:05d}-of-???.safetensors')
                save_file(current_shard, shard_path)
                print(f"Saved shard {shard_index} to {shard_path}")

                current_shard = {}
                current_size = 0
                shard_index += 1

            current_shard[key] = value
            current_size += weight_size

        # 保存最后一个分片
        if current_shard:
            shard_path = os.path.join(output_dir, f'model-{shard_index:05d}-of-???.safetensors')
            save_file(current_shard, shard_path)
            print(f"Saved shard {shard_index} to {shard_path}")

    def save_tokenizer(self, tokenizer_dir: str, output_dir: Optional[str] = None):
        """保存 tokenizer 文件"""
        if output_dir is None:
            output_dir = self.save_dir

        # 复制 tokenizer 文件
        if os.path.exists(tokenizer_dir):
            shutil.copy2(
                os.path.join(tokenizer_dir, 'tokenizer.model'),
                os.path.join(output_dir, 'tokenizer.model')
            )
            shutil.copy2(
                os.path.join(tokenizer_dir, 'tokenizer.vocab'),
                os.path.join(output_dir, 'tokenizer.vocab')
            )
            shutil.copy2(
                os.path.join(tokenizer_dir, 'tokenizer.py'),
                os.path.join(output_dir, 'tokenizer.py')
            )

            # 保存 tokenizer_config.json
            tokenizer_config = {
                "add_bos_token": True,
                "add_eos_token": True,
                "add_prefix_space": False,
                "added_tokens_decoder": {
                    "0": {"content": "<pad>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                    "1": {"content": "<unk>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                    "2": {"content": "<s>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                    "3": {"content": "</s>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
                },
                "bos_token": "<s>",
                "chat_template": "{% for message in messages %}{{ '<s>' + message.role + '\\n' + message.content + '</s>' }}{% endfor %}{{ '<s>assistant\\n' }}",
                "clean_up_tokenization_spaces": False,
                "eos_token": "</s>",
                "legacy": True,
                "model_max_length": 4096,
                "pad_token": "<pad>",
                "sp_model_kwargs": {},
                "tokenizer_class": "SentencePieceTokenizer",
                "unk_token": "<unk>"
            }

            config_path = os.path.join(output_dir, 'tokenizer_config.json')
            with open(config_path, 'w') as f:
                json.dump(tokenizer_config, f, indent=2)

            print(f"Tokenizer files saved to {output_dir}")


# =============================================================================
# 完整转换流程
# =============================================================================
def convert_to_hf(
    model: LlamaForCausalLM,
    config: ModelConfig,
    save_dir: str,
    tokenizer_dir: Optional[str] = None
):
    """
    完整转换流程

    参数:
        model: 训练好的模型
        config: 模型配置
        save_dir: 保存目录
        tokenizer_dir: tokenizer 文件目录
    """
    print(f"Converting model to HuggingFace format...")
    print(f"Save directory: {save_dir}")

    # 创建转换器
    converter = ModelConverter(config, save_dir)

    # 1. 保存 config.json
    print("Saving config...")
    converter.save_config()

    # 2. 保存模型权重
    print("Saving model weights...")
    state_dict = model.state_dict()
    converter.save_model(state_dict)

    # 3. 保存 tokenizer
    if tokenizer_dir:
        print("Saving tokenizer...")
        converter.save_tokenizer(tokenizer_dir)

    print("Conversion completed!")
    print(f"Model saved to: {save_dir}")


# =============================================================================
# 从检查点转换
# =============================================================================
def convert_from_checkpoint(
    checkpoint_path: str,
    save_dir: str,
    tokenizer_dir: Optional[str] = None,
    is_vlm: bool = False
):
    """
    从检查点转换

    参数:
        checkpoint_path: 检查点路径
        save_dir: 保存目录
        tokenizer_dir: tokenizer 目录
        is_vlm: 是否是多模态模型
    """
    # 加载检查点
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 获取配置
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = ModelConfig.from_dict(config_dict)
    else:
        config = ModelConfig()

    # 创建模型
    print("Creating model...")
    if is_vlm:
        from modeling_vlm import LlamaForConditionalGeneration
        model = LlamaForConditionalGeneration(config)
    else:
        model = LlamaForCausalLM(config)

    # 加载权重
    print("Loading weights...")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)

    # 转换
    convert_to_hf(model, config, save_dir, tokenizer_dir)


# =============================================================================
# 主函数
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert model to HuggingFace format")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint")
    parser.add_argument("--save_dir", type=str, required=True,
                       help="Save directory")
    parser.add_argument("--tokenizer_dir", type=str, default=None,
                       help="Tokenizer directory")
    parser.add_argument("--is_vlm", action="store_true",
                       help="Is multi-modal model")

    args = parser.parse_args()

    convert_from_checkpoint(
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
        tokenizer_dir=args.tokenizer_dir,
        is_vlm=args.is_vlm
    )

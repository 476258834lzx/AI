"""
模型配置模块

定义模型配置类，适配 HuggingFace 格式
支持纯文本模型和多模态模型配置
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# =============================================================================
# 视觉编码器配置
# =============================================================================
@dataclass
class VisionConfig:
    """视觉编码器配置"""
    image_size: int = 224          # 输入图像尺寸
    patch_size: int = 14            # Patch 大小
    hidden_size: int = 768          # 隐藏层维度
    num_hidden_layers: int = 12     # Transformer 层数
    num_attention_heads: int = 12   # 注意力头数
    intermediate_size: int = 3072   # FFN 中间层维度
    dropout: float = 0.0            # Dropout 概率
    layer_norm_eps: float = 1e-5    # LayerNorm epsilon

    # 与 CLIP 相关的配置
    in_channels: int = 3            # 输入通道数（RGB）
    projection_dim: int = 768       # 投影维度

    def to_dict(self):
        return {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'hidden_size': self.hidden_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'dropout': self.dropout,
            'layer_norm_eps': self.layer_norm_eps,
            'in_channels': self.in_channels,
            'projection_dim': self.projection_dim
        }


# =============================================================================
# 主模型配置
# =============================================================================
@dataclass
class ModelConfig:
    """主模型配置"""
    # 基础参数
    model_type: str = "llama"           # 模型类型
    architecture: str = "LlamaForCausalLM"  # 模型架构

    # 词汇表
    vocab_size: int = 50000             # 词汇表大小
    pad_token_id: int = 0               # Padding token ID
    bos_token_id: int = 2               # BOS token ID
    eos_token_id: int = 3               # EOS token ID
    unk_token_id: int = 1               # UNK token ID

    # 模型维度
    hidden_size: int = 256              # 隐藏层维度
    intermediate_size: int = 512        # FFN 中间层维度 (通常 4 * hidden_size)
    num_hidden_layers: int = 4          # Transformer 层数
    num_attention_heads: int = 4        # Query 头数
    num_key_value_heads: int = 2        # Key/Value 头数 (GQA)
    head_dim: int = 64                 # 每头维度

    # RoPE 配置
    max_position_embeddings: int = 512  # 最大位置编码长度
    rope_theta: float = 500000.0        # RoPE 基础频率
    rope_scaling: Optional[dict] = None # RoPE 缩放配置

    # 归一化
    rms_norm_eps: float = 1e-6          # RMSNorm epsilon

    # 初始化
    initializer_range: float = 0.02      # 权重初始化标准差

    # 权重共享
    share_input_output_embeddings: bool = True  # 共享 embedding 和 lm_head 权重

    # 视觉配置
    vision_config: Optional[VisionConfig] = None
    use_vision: bool = False            # 是否使用视觉编码器

    # LoRA 配置
    enable_lora: bool = False           # 是否启用 LoRA
    lora_rank: int = 8                 # LoRA 秩
    lora_alpha: float = 16.0           # LoRA 缩放因子
    lora_dropout: float = 0.0          # LoRA Dropout

    # 数据类型
    torch_dtype: str = "float32"        # 模型权重数据类型

    def __post_init__(self):
        # 自动计算 head_dim
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    def to_dict(self):
        """转换为字典"""
        config_dict = {
            'model_type': self.model_type,
            'architecture': self.architecture,
            'vocab_size': self.vocab_size,
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
            'unk_token_id': self.unk_token_id,
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'num_key_value_heads': self.num_key_value_heads,
            'head_dim': self.head_dim,
            'max_position_embeddings': self.max_position_embeddings,
            'rope_theta': self.rope_theta,
            'rope_scaling': self.rope_scaling,
            'rms_norm_eps': self.rms_norm_eps,
            'initializer_range': self.initializer_range,
            'share_input_output_embeddings': self.share_input_output_embeddings,
            'use_vision': self.use_vision,
            'enable_lora': self.enable_lora,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'torch_dtype': self.torch_dtype
        }

        if self.vision_config:
            config_dict['vision_config'] = self.vision_config.to_dict()

        return config_dict

    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        # 处理 vision_config
        vision_config = None
        if 'vision_config' in config_dict and config_dict['vision_config']:
            vision_dict = config_dict.pop('vision_config')
            vision_config = VisionConfig(**vision_dict)

        # 处理 torch_dtype
        torch_dtype = config_dict.pop('torch_dtype', 'float32')

        config = cls(**config_dict)
        config.vision_config = vision_config
        config.torch_dtype = torch_dtype

        return config


# =============================================================================
# 训练配置
# =============================================================================
@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数
    output_dir: str = "./output"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # 优化器
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # 调度器
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

    # 其他
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    seed: int = 42

    # 混合精度
    fp16: bool = False
    bf16: bool = True

    # DeepSpeed
    deepspeed: bool = False
    deepspeed_config: Optional[str] = None


# =============================================================================
# 生成配置
# =============================================================================
@dataclass
class GenerationConfig:
    """生成配置"""
    max_new_tokens: int = 100
    min_new_tokens: Optional[int] = None
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    def to_dict(self):
        return {
            'max_new_tokens': self.max_new_tokens,
            'min_new_tokens': self.min_new_tokens,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'do_sample': self.do_sample,
            'num_beams': self.num_beams,
            'repetition_penalty': self.repetition_penalty,
            'length_penalty': self.length_penalty,
            'no_repeat_ngram_size': self.no_repeat_ngram_size
        }


# =============================================================================
# 配置工厂
# =============================================================================
class ConfigFactory:
    """配置工厂，用于创建不同规模的模型配置"""

    @staticmethod
    def create_mini_model():
        """创建最小模型配置（用于测试）"""
        return ModelConfig(
            vocab_size=4096,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            enable_lora=False
        )

    @staticmethod
    def create_small_model():
        """创建小模型配置"""
        return ModelConfig(
            vocab_size=32000,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            enable_lora=True
        )

    @staticmethod
    def create_vlm_model():
        """创建视觉语言模型配置"""
        vision_config = VisionConfig(
            image_size=224,
            patch_size=14,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        return ModelConfig(
            vocab_size=50000,
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            head_dim=64,
            max_position_embeddings=4096,
            vision_config=vision_config,
            use_vision=True,
            enable_lora=True
        )


# =============================================================================
# 简化配置接口
# =============================================================================
def get_default_config(model_size: str = "mini") -> ModelConfig:
    """
    获取默认配置

    参数:
        model_size: "mini", "small", 或 "vlm"
    """
    factory_map = {
        "mini": ConfigFactory.create_mini_model,
        "small": ConfigFactory.create_small_model,
        "vlm": ConfigFactory.create_vlm_model
    }

    if model_size not in factory_map:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(factory_map.keys())}")

    return factory_map[model_size]()

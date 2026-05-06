"""
Storier 模型修复版本，支持 vLLM serve 调用

修复内容：
1. 移除 PreTrainedModel 继承，使 vLLM 使用原生实现而非 transformers backend
2. 使用 PretrainedConfig 作为配置基类（仅用于配置解析）
3. 添加 _supports_attention_backend 类属性
"""
import torch
from torch import nn
from typing import Iterable, Tuple, Set, Any

# transformers 相关
from transformers import PretrainedConfig

# VLLM 核心配置类
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

# VLLM 分布式相关（获取张量并行世界大小）
from vllm.distributed import get_tensor_model_parallel_world_size

# VLLM 模型层组件
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
    ColumnParallelLinear
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)

# VLLM 模型工具函数
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix

# VLLM 模型接口
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP

# VLLM 序列相关（IntermediateTensors 用于流水线并行）
from vllm.sequence import IntermediateTensors

# VLLM 模型注册
from vllm.model_executor.models import ModelRegistry


class StorierConfig(PretrainedConfig):
    """Storier 模型配置类"""
    model_type = "storier"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = kwargs.get("num_layers", 48)
        self.input_dim = kwargs.get("input_dim", 768)
        self.hide_dim = kwargs.get("hide_dim", 3072)
        self.n_q_heads = kwargs.get("n_q_heads", 12)
        self.n_kv_heads = kwargs.get("n_kv_heads", 2)
        self.max_pos_len = kwargs.get("max_pos_len", 16384)
        self.vocab_size = kwargs.get("vocab_size", 50000)
        self.use_cache = kwargs.get("use_cache", True)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 50000.0)

        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.bos_token_id = kwargs.get("bos_token_id", 2)
        self.eos_token_id = kwargs.get("eos_token_id", 3)

    @property
    def num_hidden_layers(self):
        return self.num_layers

    @property
    def hidden_size(self):
        return self.input_dim

    @property
    def intermediate_size(self):
        return self.hide_dim

    @property
    def num_attention_heads(self):
        return self.n_q_heads

    @property
    def num_key_value_heads(self):
        return self.n_kv_heads

    @property
    def max_position_embeddings(self):
        return self.max_pos_len


class StorierAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rep = self.total_num_heads // self.total_num_kv_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = ColumnParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rope_parameters = {
            "rope_type": "default",
            "theta": 50000.0,
        }
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        attn_output = attn_output.reshape(
            -1,
            self.total_num_kv_heads,
            self.rep,
            self.head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(-1, self.total_num_kv_heads * self.rep * self.head_dim)

        output, _ = self.o_proj(attn_output)
        return output


class StorierMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class StorierDecoderLayer(nn.Module):
    def __init__(
        self,
        config: StorierConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = StorierAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = StorierMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class StorierModel(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.layers = nn.ModuleList([
            StorierDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params = set()

        for name, loaded_weight in weights:
            orig_name = name

            if name == "emb.weight":
                name = "embed_tokens.weight"
            elif name == "emb":
                name = "embed_tokens"
            else:
                name = name.replace("tf_layer._layers.", "layers.")
                name = name.replace("._att_layer._qw", ".self_attn.qkv_proj")
                name = name.replace("._att_layer._kw", ".self_attn.qkv_proj")
                name = name.replace("._att_layer._vw", ".self_attn.qkv_proj")
                name = name.replace("._att_layer._ow", ".self_attn.o_proj")
                name = name.replace("._ffn_layer._w0", ".mlp.gate_up_proj")
                name = name.replace("._ffn_layer._w1", ".mlp.gate_up_proj")
                name = name.replace("._ffn_layer._w2", ".mlp.down_proj")
                name = name.replace("._att_norm._w", ".input_layernorm.weight")
                name = name.replace("._ffn_norm._w", ".post_attention_layernorm.weight")
                name = name.replace("tf_layer._out_norm._w", "norm.weight")
                name = name.lstrip("_")

            if name not in params_dict:
                print(f"Warning: Parameter {name} not found in model (from {orig_name})")
                continue

            param = params_dict[name]

            if "qkv_proj" in name and ("._qw" in orig_name or ".q_proj" in orig_name or
                                       "._kw" in orig_name or ".k_proj" in orig_name or
                                       "._vw" in orig_name or ".v_proj" in orig_name):
                if "._qw" in orig_name or "q_proj" in orig_name:
                    shard_id = "q"
                elif "._kw" in orig_name or "k_proj" in orig_name:
                    shard_id = "k"
                elif "._vw" in orig_name or "v_proj" in orig_name:
                    shard_id = "v"
                else:
                    shard_id = None
                if shard_id is not None:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name)
                    continue
            elif "gate_up_proj" in name and ("._w0" in orig_name or ".up_proj" in orig_name or
                                             "._w1" in orig_name or ".gate_proj" in orig_name):
                if "._w0" in orig_name or "up_proj" in orig_name:
                    shard_id = 1
                elif "._w1" in orig_name or "gate_proj" in orig_name:
                    shard_id = 0
                else:
                    shard_id = None
                if shard_id is not None:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name)
                    continue

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class StorierForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """
    vLLM 原生 Storier 模型实现

    特点：
    - 不继承 PreTrainedModel，使 vLLM 使用原生实现
    - 支持 vLLM 的 LoRA、流水线并行等特性
    """
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # 添加这个属性以支持某些 vLLM 功能检查
    _is_pre_archived = False

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config

        self.config = config
        self.model = StorierModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        inner_loaded = self.model.load_weights(weights)
        loaded = {f"model.{name}" for name in inner_loaded}
        return loaded

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将 token IDs 转换为嵌入向量（vLLM V1 引擎要求）"""
        return self.model.embed_tokens(input_ids)


# 注册模型到 vLLM
ModelRegistry.register_model("StorierForCausalLM", StorierForCausalLM)

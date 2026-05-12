"""
Llama 核心模型实现

基于 Llama 3 架构，包含：
1. RMSNorm: 均方根归一化
2. RoPE: 旋转位置编码
3. Attention: GQA (Grouped Query Attention)
4. SwiGLU FFN: 门控线性单元前馈网络
5. LlamaDecoderLayer: 解码器层
6. LlamaModel: 完整的 Llama 模型

适配 DeepSpeed 训练和 HuggingFace 格式转换
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# =============================================================================
# RMSNorm: 均方根归一化
# =============================================================================
class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)

    与 LayerNorm 类似，但只使用 RMS（均方根）进行归一化
    公式: output = x * (rms(x)^(2*eps-1)) = x / rms(x)

    优点: 比 LayerNorm 更快，因为只需要计算 RMS，不需要计算均值

    参数:
        hidden_size: 隐藏层维度
        eps: 防止除零的小常数
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        参数:
            hidden_states: [B, seq_len, hidden_size]

        返回:
            normalized: [B, seq_len, hidden_size]
        """
        # 计算 RMS: sqrt(mean(x^2))
        # rsqrt = 1/sqrt()
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


# =============================================================================
# RoPE: 旋转位置编码
# =============================================================================
def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 500000.0,
    device: str = "cpu"
) -> torch.Tensor:
    """
    预计算旋转位置编码的频率复数向量

    生成形状为 [end, dim/2] 的复数张量，用于 RoPE

    公式: freqs_i = theta^(-2*i/dim) for i in [0, dim/2)

    参数:
        dim: 注意力头的维度（必须是偶数）
        end: 最大序列长度
        theta: 基础频率（Llama 默认 500000）
        device: 设备

    返回:
        freqs_cis: [end, dim/2] 复数张量
    """
    # 生成频率向量
    # torch.arange(0, dim, 2) = [0, 2, 4, ..., dim-2]
    # freqs = theta^(−2i/dim)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # 生成位置索引
    # t = [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=device)

    # 外积: [end, 1] * [1, dim/2] -> [end, dim/2]
    # 每个位置 t 对应一个频率向量
    freqs = torch.outer(t, freqs)

    # 转换为极坐标形式: 模长=1, 角度=freqs
    # torch.polar(abs, angle) = abs * cos(angle) + abs * sin(angle)j
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    调整 freqs_cis 形状以便广播到 x

    参数:
        freqs_cis: [seq_len, head_dim/2]
        x: [B, num_heads, seq_len, head_dim]

    返回:
        freqs_cis: [1, 1, seq_len, head_dim/2] 用于广播
    """
    # 只取前 seq_len 个位置
    freqs_cis = freqs_cis[:x.shape[2]]
    # 扩展维度以便广播: [seq_len, dim/2] -> [1, 1, seq_len, dim/2]
    return freqs_cis[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    旋转半部分维度

    将向量 [x1, x2, x3, x4, ...] 转换为 [-x3, -x4, x1, x2, ...]

    用于 RoPE 的核心操作

    参数:
        x: [B, num_heads, seq_len, head_dim]

    返回:
        rotated: [B, num_heads, seq_len, head_dim]
    """
    # 将最后一维分成两半
    x1 = x[..., : x.shape[-1] // 2]  # 前半部分
    x2 = x[..., x.shape[-1] // 2:]   # 后半部分
    # 旋转: [-x2, x1]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码到 Q 和 K

    公式: RoPE(x) = x * cos(theta) + rotate(x) * sin(theta)

    参数:
        q: [B, num_heads, seq_len, head_dim]
        k: [B, num_kv_heads, seq_len, head_dim]
        freqs_cis: [seq_len, head_dim/2] 复数张量

    返回:
        q_rot: [B, num_heads, seq_len, head_dim]
        k_rot: [B, num_kv_heads, seq_len, head_dim]
    """
    # 调整形状以便广播
    freqs_cis = reshape_for_broadcast(freqs_cis, q)

    # 将实数张量转换为复数表示
    # [B, num_heads, seq_len, head_dim] -> [B, num_heads, seq_len, head_dim/2, 2]
    # 然后转换为复数 [B, num_heads, seq_len, head_dim/2]
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # 复数乘法: [B, num_heads, seq_len, head_dim/2] * [1, 1, seq_len, head_dim/2]
    # 广播后应用旋转
    q_floats = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
    k_floats = torch.view_as_real(k_complex * freqs_cis).flatten(-2)

    return q_floats.type_as(q), k_floats.type_as(k)


# =============================================================================
# LoRA Linear: 低秩适应（作为类内函数）
# =============================================================================
class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 线性层

    在预训练权重旁边添加低秩分解矩阵:
        y = Wx + BAx * (alpha/rank)

    其中:
        W: 原始冻结权重
        B, A: 可训练的低秩矩阵 (rank << hidden_dim)
        alpha: 缩放因子

    参数:
        in_features: 输入维度
        out_features: 输出维度
        rank: LoRA 秩
        alpha: 缩放因子
        dropout: Dropout 概率
        bias: 是否使用偏置
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 原始权重（冻结）
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 初始化
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [B, *, in_features]

        返回:
            output: [B, *, out_features]
        """
        # 原始输出
        base_output = F.linear(x, self.weight, self.bias)

        # LoRA 输出
        # lora_A: [rank, in_features], x: [B, seq, in_features]
        # lora_A @ x^T: [rank, in_features] @ [B, seq, in_features]^T
        # -> [rank, B, seq] -> [B, rank, seq]
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T

        return base_output + lora_output * self.scaling

    def merge_weights(self):
        """合并 LoRA 权重到原始权重"""
        self.weight.data += self.lora_B @ self.lora_A * self.scaling

    def unmerge_weights(self):
        """解合并，恢复原始权重"""
        self.weight.data -= self.lora_B @ self.lora_A * self.scaling


# =============================================================================
# Attention: GQA (Grouped Query Attention)
# =============================================================================
class LlamaAttention(nn.Module):
    """
    Llama 多头注意力，支持 GQA (Grouped Query Attention)

    GQA 优化: 只计算 n_kv_heads 个 Key/Value，复制到所有 Query heads
    减少 KV 缓存大小，加速推理

    参数:
        hidden_size: 隐藏层维度
        num_attention_heads: Query 头数
        num_key_value_heads: Key/Value 头数（< num_attention_heads 时启用 GQA）
        head_dim: 每个头的维度
        dropout: Dropout 概率
        bias: 是否使用偏置
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.group = num_attention_heads // num_key_value_heads

        # QKV 投影
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=bias)

        self.dropout = dropout
        self.is_gqa = self.group > 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> torch.Tensor:
        """
        参数:
            hidden_states: [B, seq_len, hidden_size]
            freqs_cis: [max_seq_len, head_dim/2] 预计算的 RoPE 频率
            position_ids: [B, seq_len] 位置索引
            attention_mask: [B, 1, seq_len, seq_len] 注意力掩码
            is_training: 是否训练模式

        返回:
            output: [B, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV 投影
        # [B, seq_len, hidden_size] -> [B, seq_len, num_heads * head_dim]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 重塑为多头格式
        # [B, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # GQA: 将 KV 头复制到 Q 头（在应用 RoPE 之前）
        if self.is_gqa:
            # [B, seq_len, num_kv_heads, head_dim] -> [B, seq_len, num_heads, head_dim]
            # 使用 repeat_interleave 在 kv_heads 维度上重复
            k = k.repeat_interleave(self.group, dim=2)
            v = v.repeat_interleave(self.group, dim=2)

        # 应用 RoPE 位置编码
        q, k = apply_rotary_pos_emb(q, k, freqs_cis)

        # 维度重排: [B, seq_len, num_heads, head_dim] -> [B, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 使用 PyTorch 内置的 scaled_dot_product_attention
        # 自动处理 causal mask 和 dropout
        if attention_mask is not None:
            # attention_mask: [B, 1, seq_len, seq_len] 或 [B, 1, 1, seq_len]
            attn_mask = attention_mask
        else:
            attn_mask = None

        # is_causal=True 自动生成 causal mask
        is_causal = attention_mask is None and not is_training

        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if is_training else 0.0,
            is_causal=is_causal
        )

        # 维度恢复: [B, num_heads, seq_len, head_dim] -> [B, seq_len, num_heads * head_dim]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)

        # 输出投影
        output = self.o_proj(output)

        return output


# =============================================================================
# SwiGLU FFN: 门控线性单元前馈网络
# =============================================================================
class LlamaMLP(nn.Module):
    """
    SwiGLU 前馈网络

    结构: Gate Projection -> SiLU/Gate -> Up Projection -> Down Projection

    与标准 FFN 的区别:
    - 使用 SwiGLU 激活: SiLU(x * sigmoid(x)) 而非 ReLU
    - 三个线性层 (gate, up, down) 而非两个 (up, down)

    参数:
        hidden_size: 隐藏层维度
        intermediate_size: 中间层维度（通常 4 * hidden_size）
        bias: 是否使用偏置
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        参数:
            hidden_states: [B, seq_len, hidden_size]

        返回:
            output: [B, seq_len, hidden_size]
        """
        # gate = W_gate(x), up = W_up(x)
        gate = self.activation_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        # SwiGLU: gate * up (逐元素乘法)
        return self.down_proj(gate * up)


# =============================================================================
# Decoder Layer: 解码器层
# =============================================================================
class LlamaDecoderLayer(nn.Module):
    """
    Llama 解码器层

    结构（Llama 3）:
        input_layernorm -> attention -> post_attention_layernorm -> mlp -> residue

    参数:
        hidden_size: 隐藏层维度
        num_attention_heads: Query 头数
        num_key_value_heads: Key/Value 头数
        head_dim: 每头维度
        intermediate_size: FFN 中间层维度
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Input LayerNorm
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Self Attention
        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim
        )

        # Post Attention LayerNorm
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # MLP
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> torch.Tensor:
        """
        参数:
            hidden_states: [B, seq_len, hidden_size]
            freqs_cis: RoPE 频率
            position_ids: 位置索引
            attention_mask: 注意力掩码

        返回:
            output: [B, seq_len, hidden_size]
        """
        # 保存残差
        residual = hidden_states

        # Input LayerNorm
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            freqs_cis,
            position_ids,
            attention_mask,
            is_training
        )

        # 残差连接
        hidden_states = residual + hidden_states
        residual = hidden_states

        # Post Attention LayerNorm
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # 残差连接
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Llama Model: 完整的 Llama 模型
# =============================================================================
class LlamaModel(nn.Module):
    """
    Llama 因果语言模型

    结构:
        embedding -> layers -> norm -> logits

    参数:
        config: LlamaConfig 配置对象
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers

        # Token Embedding
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx
        )

        # Decoder Layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                rms_norm_eps=config.rms_norm_eps
            )
            for _ in range(config.num_hidden_layers)
        ])

        # Final LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 频率
        self.freqs_cis = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            theta=config.rope_theta,
            device='cpu'
        )

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """权重初始化"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ):
        """
        参数:
            input_ids: [B, seq_len] token IDs
            attention_mask: [B, seq_len] 注意力掩码
            position_ids: [B, seq_len] 位置索引

        返回:
            hidden_states: [B, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Token Embedding
        hidden_states = self.embed_tokens(input_ids)

        # 获取 RoPE 频率（移动到正确设备）
        freqs_cis = self.freqs_cis.to(hidden_states.device)
        if position_ids is not None:
            freqs_cis = freqs_cis[position_ids]
        else:
            freqs_cis = freqs_cis[:seq_len]

        # Decoder Layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                freqs_cis,
                position_ids,
                attention_mask
            )

        # Final LayerNorm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_k=50):
        """简单的自回归生成"""
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        for _ in range(max_new_tokens):
            # 前向传播（限制上下文）
            hidden_states = self.forward(input_ids)

            # 只取最后一个 token 的输出
            logits = hidden_states[:, -1, :]

            # 应用 temperature
            if temperature > 0:
                logits = logits / temperature

            # Top-k 采样
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# =============================================================================
# LlamaForCausalLM: 因果语言模型
# =============================================================================
class LlamaForCausalLM(nn.Module):
    """
    Llama 因果语言模型（带语言建模头）

    参数:
        config: LlamaConfig 配置对象
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # 基础模型
        self.model = LlamaModel(config)

        # LM Head（可选：与 embedding 共享权重）
        if config.share_input_output_embeddings:
            # 与 embedding 共享权重
            self.lm_head = lambda x: x @ self.model.embed_tokens.weight.T
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            # 权重初始化
            nn.init.normal_(self.lm_head.weight, std=config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ):
        """
        参数:
            input_ids: [B, seq_len] token IDs
            labels: [B, seq_len] 目标 token IDs（用于计算损失）
            attention_mask: [B, seq_len] 注意力掩码
            position_ids: [B, seq_len] 位置索引

        返回:
            dict: {'logits': logits, 'loss': loss, 'hidden_states': hidden_states}
        """
        # 基础模型
        hidden_states = self.model(
            input_ids,
            attention_mask,
            position_ids
        )

        # LM Head
        if self.config.share_input_output_embeddings:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 计算交叉熵损失
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Flatten
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        eos_token_id: int = 2
    ):
        """
        自回归生成

        参数:
            input_ids: [B, seq_len] 输入 token IDs
            max_new_tokens: 最大生成 token 数
            temperature: 温度（0 表示贪心）
            top_k: Top-k 采样
            eos_token_id: 结束 token ID

        返回:
            output_ids: [B, seq_len + new_tokens] 生成的 token IDs
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        for _ in range(max_new_tokens):
            # 限制输入长度（KV 缓存优化）
            model_inputs = input_ids

            # 前向传播
            hidden_states = self.model(model_inputs)

            # LM Head
            if self.config.share_input_output_embeddings:
                logits = self.lm_head(hidden_states)
            else:
                logits = self.lm_head(hidden_states)

            # 只取最后一个 token
            next_token_logits = logits[:, -1, :]

            # Temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Top-k
            if top_k > 0:
                top_k_val = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 结束检查
            if (next_token == eos_token_id).all():
                break

        return input_ids

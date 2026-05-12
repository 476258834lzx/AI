"""
视觉编码器模块：自定义 CLIP ViT 实现

本模块提供两种视觉编码器实现：
1. CustomCLIPVisionEncoder: 自定义实现的 CLIP Vision Transformer
2. TorchVisionViT 参考: torchvision 官方实现的用法对照

用法对照:
    # torchvision 实现（参考）
    # from torchvision.models import vision_transformer, ViT_B_16_Weights
    # model = vision_transformer.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    # # 输入: [B, 3, 224, 224] -> 输出: [B, 768] (CLS token)
    # image = torch.randn(1, 3, 224, 224)
    # features = model(image)  # [1, 768]

    # 自定义实现
    # from vision_encoder import CustomCLIPVisionEncoder
    # encoder = CustomCLIPVisionEncoder(
    #     image_size=224,
    #     patch_size=14,
    #     hidden_size=768,
    #     num_hidden_layers=12,
    #     num_attention_heads=12,
    #     intermediate_size=3072
    # )
    # features = encoder(pixel_values)  # [B, hidden_size]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """
    图像分块嵌入层

    将输入图像分割成固定大小的 patches，并线性投影到 embedding 维度
    添加 [CLS] token 和位置编码

    参数:
        image_size: 输入图像尺寸（假设为正方形）
        patch_size: 每个 patch 的尺寸
        in_channels: 输入通道数（RGB 为 3）
        hidden_size: 嵌入维度
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        hidden_size: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 例如 224/14=16, 16*16=256

        # 卷积实现 patch embedding（等价于 flatten + linear）
        # 每个 patch 被展平并投影到 hidden_size
        self.projection = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

        # [CLS] token：可学习的分类 token，拼接在 patch tokens 前面
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # 位置编码：每个 patch + 1 个 CLS token
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        参数:
            pixel_values: [B, C, H, W] 格式的图像张量

        返回:
            embeddings: [B, num_patches+1, hidden_size] patch embeddings 加上 CLS token
            original_shape: 用于重建的原始形状信息
        """
        batch_size = pixel_values.shape[0]
        original_shape = pixel_values.shape  # [B, C, H, W]

        # [B, C, H, W] -> [B, hidden_size, num_patches_h, num_patches_w]
        # 例如: [B, 3, 224, 224] -> [B, 768, 16, 16]
        patch_embeddings = self.projection(pixel_values)

        # [B, hidden_size, H', W'] -> [B, num_patches, hidden_size]
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)

        # [1, 1, hidden_size] -> [B, 1, hidden_size]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # 拼接 CLS token
        # [B, num_patches, hidden_size] + [B, 1, hidden_size] -> [B, num_patches+1, hidden_size]
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)

        # 添加位置编码
        # position_ids: [B, num_patches+1] = [[0, 1, 2, ..., num_patches], ...]
        position_ids = torch.arange(self.num_positions, device=pixel_values.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        embeddings = embeddings + self.position_embedding(position_ids)

        return embeddings, original_shape


class CLIPAttention(nn.Module):
    """
    CLIP 多头自注意力机制

    包含完整的 QKV 投影和输出投影，支持 dropout
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.scale = self.head_size ** -0.5

        assert hidden_size % num_attention_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"

        # QKV 投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        参数:
            hidden_states: [B, seq_len, hidden_size]
            attention_mask: [B, seq_len] 可选的注意力掩码

        返回:
            context: [B, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV 投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 重塑为多头格式
        # [B, seq_len, hidden_size] -> [B, num_heads, seq_len, head_size]
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)

        # 计算注意力分数
        # [B, num_heads, seq_len, head_size] @ [B, num_heads, head_size, seq_len]
        # -> [B, num_heads, seq_len, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            # 注意力掩码: [B, seq_len] -> [B, 1, 1, seq_len] 用于广播
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        # Softmax + Dropout
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 应用注意力到值
        # [B, num_heads, seq_len, seq_len] @ [B, num_heads, seq_len, head_size]
        # -> [B, num_heads, seq_len, head_size]
        context = torch.matmul(attn_probs, v)

        # 合并多头
        # [B, num_heads, seq_len, head_size] -> [B, seq_len, hidden_size]
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)

        # 输出投影
        output = self.out_proj(context)

        return output


class CLIPMLP(nn.Module):
    """
    CLIP 前馈网络

    结构: Linear -> GELU -> Linear
    使用 GELU 激活函数（CLIP 官方实现）
    """
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        dropout: float = 0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    """
    CLIP 编码器层

    结构: LayerNorm -> Attention -> LayerNorm -> MLP -> Residue
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = CLIPAttention(hidden_size, num_attention_heads, dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = CLIPMLP(hidden_size, intermediate_size, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 自注意力 + 残差连接
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # 前馈网络 + 残差连接
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """
    CLIP 编码器：由多个编码器层组成

    参数:
        num_hidden_layers: 编码器层数
        hidden_size: 隐藏层维度
        num_attention_heads: 注意力头数
        intermediate_size: 前馈网络中间层维度
    """
    def __init__(
        self,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                dropout,
                layer_norm_eps
            )
            for _ in range(num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class CustomCLIPVisionEncoder(nn.Module):
    """
    自定义 CLIP Vision Encoder

    完整的视觉编码器，包含：
    1. Patch Embedding: 将图像分割成 patches 并嵌入
    2. Transformer Encoder: 多层自注意力
    3. LayerNorm: 最终归一化

    输出 [CLS] token 作为图像特征表示

    用法:
        encoder = CustomCLIPVisionEncoder(
            image_size=224,
            patch_size=14,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        # pixel_values: [B, 3, 224, 224]
        features = encoder(pixel_values)  # [B, hidden_size]
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        # Patch Embedding 层
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            hidden_size=hidden_size
        )

        # Transformer 编码器
        self.encoder = CLIPEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

        # 最终 LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False
    ) -> dict:
        """
        前向传播

        参数:
            pixel_values: [B, 3, H, W] 格式的图像张量
            output_hidden_states: 是否返回所有层的隐藏状态

        返回:
            dict: {
                'last_hidden_state': [B, num_patches+1, hidden_size],
                'pooler_output': [B, hidden_size],  # CLS token
                'hidden_states': Optional[List[B, num_patches+1, hidden_size]] 所有层输出
            }
        """
        # Patch embedding + position encoding
        # [B, 3, H, W] -> [B, num_patches+1, hidden_size]
        hidden_states, original_shape = self.patch_embedding(pixel_values)

        # Transformer 编码器
        # [B, num_patches+1, hidden_size]
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        hidden_states = self.encoder(hidden_states)

        # 最终归一化
        last_hidden_state = self.layer_norm(hidden_states)

        # 提取 [CLS] token 作为图像表示
        # [B, num_patches+1, hidden_size] -> [B, hidden_size]
        pooler_output = last_hidden_state[:, 0, :]

        return {
            'last_hidden_state': last_hidden_state,
            'pooler_output': pooler_output,
            'hidden_states': all_hidden_states
        }


# torchvision 实现参考（保留用于对照）
# ======================================
"""
# torchvision 官方 CLIP ViT 实现用法示例

from torchvision.models import vision_transformer, CLIPVisionModel, CLIPVisionModel_Weights
import torch

# 方式1: 使用预训练 CLIP Vision Model
model = CLIPVisionModel(weights=CLIPVisionModel_Weights.DEFAULT)
model.eval()

# 输入预处理（需要与训练时相同的归一化）
# 预处理: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
preprocess = CLIPVisionModel_Weights.DEFAULT.transforms()

# 输入图像 [B, 3, 224, 224]
image = torch.randn(1, 3, 224, 224)

# 前向传播
with torch.no_grad():
    output = model(image)
    # output.last_hidden_state: [B, 257, 768] (256 patches + 1 CLS)
    # output.pooler_output: [B, 768]

# 方式2: 基础 ViT 实现
model = vision_transformer.vit_b_16(weights=None)
# 参数对照:
# - image_size: 224 (默认)
# - patch_size: 16 (默认)
# - num_classes: 1000 (分类头，不用于 CLIP)
# - hidden_dim: 768
# - num_heads: 12
# - num_layers: 12
# - mlp_dim: 3072

# 获取特征（移除分类头）
model.heads.head = nn.Identity()
features = model(image)  # [B, 768]

# 方式3: 自定义参数
model = vision_transformer.vit_b_16(
    image_size=336,  # Qwen-VL 使用 448
    patch_size=14,    # ViT-B/14
    num_classes=512,  # 自定义输出维度
    hidden_dim=768,
    num_heads=12,
    num_layers=12,
    mlp_dim=3072
)
"""

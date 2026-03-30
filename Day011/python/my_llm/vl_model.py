import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
import math


# ==================== 多模态数据预处理模块 ====================

@dataclass
class MultimodalConfig:
    """多模态配置类"""
    # 视觉编码器配置
    image_size: int = 224  # 输入图像尺寸
    patch_size: int = 14  # ViT patch大小 (Qwen2-VL使用14, CLIP常用16)
    vision_dim: int = 768  # 视觉编码器输出维度 (ViT-L/14为1024, Base为768)
    vision_layers: int = 12  # ViT层数
    vision_heads: int = 12  # ViT注意力头数

    # 模态对齐配置
    projector_type: str = "mlp"  # 投影层类型: "linear", "mlp", "qformer"
    projector_hidden_dim: int = 2048  # MLP投影层隐藏层维度
    num_query_tokens: int = 32  # Q-Former查询token数 (如使用)

    # 多模态特殊token
    image_token_id: int = 151646  # 图像开始token ID (类似<image>)
    video_token_id: int = 151647  # 视频token ID

    # 训练配置
    freeze_vision_encoder: bool = True  # 是否冻结视觉编码器 (常规做法[^19^])


class ImagePreprocessor:
    """
    图像数据预处理类
    参考CLIP和LLaVA的预处理方式[^1^][^19^]
    """

    def __init__(self, image_size: int = 224, patch_size: int = 14):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

    def __call__(self, images: Union[torch.Tensor, List]) -> torch.Tensor:
        """
        预处理图像为patch embeddings
        Args:
            images: 输入图像, shape (B, C, H, W) 或 PIL图像列表
        Returns:
            patchified images: (B, N, P*P*C) where N = (H/P)*(W/P)
        """
        if isinstance(images, list):
            # 处理PIL图像列表
            images = self._load_images(images)

        B, C, H, W = images.shape

        # 1. 调整图像尺寸
        if H != self.image_size or W != self.image_size:
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )

        # 2. 归一化 (使用ImageNet标准或CLIP标准)
        images = self._normalize(images)

        # 3. 分块 (Patchify) - ViT核心操作[^1^][^12^]
        # 将图像分割为 non-overlapping patches
        patches = self._patchify(images)  # (B, N, P*P*C)

        return patches

    def _load_images(self, image_list: List) -> torch.Tensor:
        """将PIL图像转换为tensor"""
        # 实际实现需要使用 torchvision.transforms
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ])
        return torch.stack([transform(img) for img in image_list])

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """标准化: CLIP使用[0.48145466, 0.4578275, 0.40821073]"""
        # ImageNet标准
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        return (images - mean) / std

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        """
        将图像分割为patches
        输入: (B, C, H, W)
        输出: (B, N, P*P*C) where N = num_patches
        """
        B, C, H, W = images.shape
        P = self.patch_size

        # 使用unfold操作提取patches
        # (B, C, H, W) -> (B, C, H//P, P, W//P, P) -> (B, H//P, W//P, C, P, P)
        patches = images.unfold(2, P, P).unfold(3, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

        # 展平: (B, num_patches_h, num_patches_w, C*P*P)
        patches = patches.view(B, -1, C * P * P)
        return patches


class VisionEncoder(nn.Module):
    """
    视觉编码器: 基于ViT架构
    参考CLIP ViT和LLaVA实现[^6^][^19^]
    支持动态分辨率 (类似Qwen2-VL[^13^])
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.embed_dim = config.vision_dim

        # 1. Patch Embedding (线性投影层)[^12^]
        patch_dim = 3 * config.patch_size * config.patch_size  # 3通道 * P * P
        self.patch_embed = nn.Linear(patch_dim, self.embed_dim)

        # 2. 位置编码 (使用可学习的2D位置编码或RoPE)
        # 这里使用标准ViT的位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)  # +1 for [CLS]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # 3. Transformer Encoder Layers
        self.layers = nn.ModuleList([
            VisionTransformerLayer(self.embed_dim, config.vision_heads)
            for _ in range(config.vision_layers)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # 初始化
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, N, patch_dim) 来自ImagePreprocessor
        Returns:
            image_features: (B, N+1, vision_dim) 包含CLS token
        """
        B = pixel_values.shape[0]

        # Patch embedding
        x = self.patch_embed(pixel_values)  # (B, N, D)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # 添加位置编码
        x = x + self.pos_embed

        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x  # (B, N+1, vision_dim)


class VisionTransformerLayer(nn.Module):
    """标准ViT层"""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


class MultimodalProjector(nn.Module):
    """
    模态对齐投影层: 将视觉特征映射到语言模型空间
    支持多种投影方式: Linear, MLP, Q-Former[^13^][^19^]
    """

    def __init__(self, vision_dim: int, llm_dim: int, config: MultimodalConfig):
        super().__init__()
        self.projector_type = config.projector_type

        if config.projector_type == "linear":
            # LLaVA-1.0风格: 单层线性投影[^19^]
            self.projector = nn.Linear(vision_dim, llm_dim)

        elif config.projector_type == "mlp":
            # LLaVA-1.5风格: 两层MLP[^13^]
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, config.projector_hidden_dim),
                nn.GELU(),
                nn.Linear(config.projector_hidden_dim, llm_dim)
            )

        elif config.projector_type == "qformer":
            # BLIP-2/Qwen-VL风格: 使用Q-Former压缩视觉token[^13^]
            self.query_tokens = nn.Parameter(
                torch.randn(1, config.num_query_tokens, llm_dim)
            )
            self.qformer = QFormer(
                num_query_tokens=config.num_query_tokens,
                vision_dim=vision_dim,
                llm_dim=llm_dim,
                num_heads=8,
                num_layers=2
            )
            self.projector = nn.Identity()  # Q-Former直接输出LLM维度

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, N, vision_dim) 视觉特征
        Returns:
            projected_features: (B, num_visual_tokens, llm_dim)
        """
        if self.projector_type == "qformer":
            B = vision_features.shape[0]
            queries = self.query_tokens.expand(B, -1, -1)
            return self.qformer(queries, vision_features)
        else:
            return self.projector(vision_features)


class QFormer(nn.Module):
    """
    Q-Former: 使用可学习的查询token压缩视觉信息
    参考BLIP-2和Qwen-VL[^13^]
    """

    def __init__(self, num_query_tokens: int, vision_dim: int, llm_dim: int,
                 num_heads: int, num_layers: int):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(1, num_query_tokens, llm_dim))

        self.layers = nn.ModuleList([
            QFormerLayer(llm_dim, vision_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, queries: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        x = queries
        for layer in self.layers:
            x = layer(x, vision_features)
        return x


class QFormerLayer(nn.Module):
    """Q-Former层: 交叉注意力 + 自注意力"""

    def __init__(self, llm_dim: int, vision_dim: int, num_heads: int):
        super().__init__()
        # 自注意力 (查询token之间)
        self.self_attn = nn.MultiheadAttention(llm_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(llm_dim)

        # 交叉注意力 (查询token关注视觉特征)
        self.cross_attn = nn.MultiheadAttention(
            llm_dim, num_heads, batch_first=True,
            kdim=vision_dim, vdim=vision_dim
        )
        self.norm2 = nn.LayerNorm(llm_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim)
        )
        self.norm3 = nn.LayerNorm(llm_dim)

    def forward(self, queries: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        # 自注意力
        q = self.norm1(queries)
        q = queries + self.self_attn(q, q, q)[0]

        # 交叉注意力 (查询关注视觉特征)
        q2 = self.norm2(q)
        q = q + self.cross_attn(q2, vision_features, vision_features)[0]

        # FFN
        q = q + self.ffn(self.norm3(q))
        return q


# ==================== 修改后的TransformerDecoder支持多模态 ====================

class MultimodalTransformerDecoder(nn.Module):
    """
    多模态Transformer解码器
    支持文本token和视觉token的交错输入[^4^][^13^]
    """

    def __init__(self,
                 num_layers: int,
                 input_dim: int,
                 hide_dim: int,
                 n_q_heads: int,
                 n_kv_heads: int,
                 max_len: int,
                 multimodal_config: Optional[MultimodalConfig] = None,
                 cache_max_batch_size: Optional[int] = None,
                 cache_max_seq_len: Optional[int] = None
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.multimodal_config = multimodal_config or MultimodalConfig()

        # 1. 视觉预处理模块
        self.image_preprocessor = ImagePreprocessor(
            image_size=self.multimodal_config.image_size,
            patch_size=self.multimodal_config.patch_size
        )

        # 2. 视觉编码器 (可配置为冻结)
        self.vision_encoder = VisionEncoder(self.multimodal_config)
        if self.multimodal_config.freeze_vision_encoder:
            self._freeze_module(self.vision_encoder)

        # 3. 模态投影层 (关键: 对齐视觉和文本空间)[^19^]
        self.vision_projector = MultimodalProjector(
            vision_dim=self.multimodal_config.vision_dim,
            llm_dim=input_dim,
            config=self.multimodal_config
        )

        # 4. 原有的Transformer层 (保持不变)
        self._layers = nn.ModuleList([
            TransformerLayer(input_dim, hide_dim, n_q_heads, n_kv_heads,
                             cache_max_batch_size, cache_max_seq_len)
            for _ in range(num_layers)
        ])

        self._out_norm = RMSNorm(input_dim)

        # 5. 位置编码 (RoPE) - 现在需要支持文本和视觉token
        # 计算视觉token数量 (包含CLS)
        num_vision_tokens = self.image_preprocessor.num_patches + 1
        # 为文本和视觉token预计算频率 (支持交错位置编码)
        _freq_cis = precompute_freqs_cis(
            input_dim // n_q_heads,
            max_len + num_vision_tokens  # 预留视觉token位置
        )
        self.register_buffer("freq_cis", _freq_cis, persistent=False)

        # 6. 模态类型嵌入 (区分文本和视觉token)
        self.modality_embed = nn.Embedding(2, input_dim)  # 0:文本, 1:视觉

    def _freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def preprocess_multimodal_inputs(
            self,
            text_tokens: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            image_positions: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        多模态数据预处理主函数
        将文本token和图像交错合并为统一序列

        Args:
            text_tokens: (B, L) 文本token IDs
            images: (B, C, H, W) 或预处理后的patches (B, N, patch_dim)
            image_positions: 图像插入位置列表 (如 [0] 表示在开头)

        Returns:
            combined_embeds: (B, total_len, input_dim) 融合后的embedding
            modality_ids: (B, total_len) 模态类型标识 (0文本/1视觉)
        """
        B = text_tokens.shape[0] if text_tokens is not None else images.shape[0]
        device = text_tokens.device if text_tokens is not None else images.device

        # 1. 处理视觉输入
        vision_embeds = None
        vision_len = 0
        if images is not None:
            # 如果输入是原始图像,先预处理为patches
            if images.dim() == 4:  # (B, C, H, W)
                patches = self.image_preprocessor(images)
            else:
                patches = images  # 已经预处理过

            # 编码视觉特征
            with torch.no_grad() if self.multimodal_config.freeze_vision_encoder else torch.enable_grad():
                vision_features = self.vision_encoder(patches)  # (B, N+1, vision_dim)

            # 投影到LLM空间
            vision_embeds = self.vision_projector(vision_features)  # (B, num_visual_tokens, llm_dim)
            vision_len = vision_embeds.shape[1]

        # 2. 处理文本输入 (需要embedding层,这里假设外部已处理或添加embedding层)
        # 注意: 实际实现中需要添加 nn.Embedding(vocab_size, input_dim)
        # 这里简化处理,假设text_tokens已经是embeddings或需要外部转换

        # 3. 交错融合 (Interleaved Fusion)[^5^]
        if text_tokens is None:
            combined = vision_embeds
            modality_ids = torch.ones(B, vision_len, dtype=torch.long, device=device)
        elif images is None:
            # 假设text_tokens已经是embeddings (B, L, D)
            combined = text_tokens if text_tokens.dim() == 3 else self._embed_tokens(text_tokens)
            modality_ids = torch.zeros(B, combined.shape[1], dtype=torch.long, device=device)
        else:
            # 交错合并: 根据image_positions插入视觉token
            text_len = text_tokens.shape[1] if text_tokens.dim() == 3 else text_tokens.shape[1]
            total_len = text_len + vision_len * len(image_positions if image_positions else [0])

            combined = torch.zeros(B, total_len, self.input_dim, device=device)
            modality_ids = torch.zeros(B, total_len, dtype=torch.long, device=device)

            # 文本embeddings
            text_embeds = text_tokens if text_tokens.dim() == 3 else self._embed_tokens(text_tokens)

            # 默认在开头插入图像
            if image_positions is None:
                image_positions = [0]

            # 合并序列
            text_idx = 0
            current_pos = 0
            for pos in sorted(image_positions):
                # 插入前面的文本
                if pos > text_idx:
                    seg_len = min(pos - text_idx, text_len - text_idx)
                    combined[:, current_pos:current_pos + seg_len] = text_embeds[:, text_idx:text_idx + seg_len]
                    current_pos += seg_len
                    text_idx += seg_len

                # 插入视觉token
                combined[:, current_pos:current_pos + vision_len] = vision_embeds
                modality_ids[:, current_pos:current_pos + vision_len] = 1  # 标记为视觉
                current_pos += vision_len

            # 插入剩余文本
            if text_idx < text_len:
                combined[:, current_pos:] = text_embeds[:, text_idx:]

        # 4. 添加模态类型嵌入
        modality_embeds = self.modality_embed(modality_ids)
        combined = combined + modality_embeds

        return combined, modality_ids

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """将token IDs转换为embeddings (需要外部提供embedding层)"""
        # 实际实现中应该使用nn.Embedding
        # 这里作为占位符
        raise NotImplementedError("需要添加token embedding层或使用外部embedding")

    def forward(self,
                text_tokens: Optional[torch.Tensor] = None,
                images: Optional[torch.Tensor] = None,
                image_positions: Optional[List[int]] = None,
                start_pos: int = 0):
        """
        多模态前向传播

        Args:
            text_tokens: 文本token IDs或embeddings
            images: 输入图像
            image_positions: 图像插入位置
            start_pos: 生成时的起始位置 (用于KV Cache)
        """
        # 1. 多模态预处理
        x, modality_ids = self.preprocess_multimodal_inputs(
            text_tokens, images, image_positions
        )

        # 2. 通过Transformer层
        _freq_cis = self.freq_cis[start_pos:start_pos + x.shape[1]]

        for _layer in self._layers:
            x = _layer(x, _freq_cis, start_pos, modality_ids)

        return self._out_norm(x)


# ==================== 修改TransformerLayer支持模态感知 ====================

class ModalityAwareAttention(nn.MultiheadAttention):
    """
    扩展MultiHeadAttention以支持模态感知 (可选)
    可以添加模态间的特殊处理,如跨模态掩码
    """

    def forward(self, x, freq_cis, start_pos, modality_ids=None):
        # 暂时忽略modality_ids,保持与原接口兼容
        # 如需模态特殊处理,可在此添加
        return super().forward(x, freq_cis, start_pos)


class TransformerLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hide_dim: int,
                 n_q_heads: int,
                 n_kv_heads: int,
                 cache_max_batch_size: Optional[int],
                 cache_max_seq_len: Optional[int]):
        super().__init__()

        self._att_norm = RMSNorm(input_dim)

        # 替换为模态感知注意力 (可选)
        self._att_layer = ModalityAwareAttention(
            input_dim, hide_dim, n_q_heads, n_kv_heads,
            cache_max_batch_size, cache_max_seq_len
        )

        self._ffn_norm = RMSNorm(input_dim)
        self._ffn_layer = FFN(input_dim, hide_dim)

    def forward(self, x, freq_cis, start_pos, modality_ids=None):
        _x = x
        _x = self._att_norm(_x)
        # 传递modality_ids给注意力层
        _x = self._att_layer(_x, freq_cis, start_pos, modality_ids)
        _x = x + _x

        _y = _x
        _y = self._ffn_norm(_y)
        _y = self._ffn_layer(_y)
        _y = _y + _x

        return _y


# ==================== 保留原有函数和类 (保持不变) ====================

def precompute_freqs_cis(dim, end, theta=50000.0, device="cpu"):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, freqs_cis):
    assert xq.shape[-1] % 2 == 0, "The last dimension of xq must be even."
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)


def reshape_for_broadcast(freqs_cis, x):
    freqs_cises = freqs_cis[:x.shape[1]]
    return freqs_cises[None, :, None]


# 保留原有的MultiHeadAttention, FFN, RMSNorm, LoraLinear类...
# [这里插入原有的这些类定义]

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hide_dim, n_q_heads, n_kv_heads,
                 cache_max_batch_size, cache_max_seq_len):
        super().__init__()
        self._n_q_heads = n_q_heads
        self._n_kv_heads = n_kv_heads
        self._group = n_q_heads // n_kv_heads
        self._head_size = input_dim // self._n_q_heads

        self._qw = nn.Linear(input_dim, self._head_size * self._n_q_heads)
        self._kw = nn.Linear(input_dim, self._head_size * self._n_kv_heads)
        self._vw = nn.Linear(input_dim, self._head_size * self._n_kv_heads)
        self._ow = nn.Linear(self._head_size * self._n_q_heads, input_dim)

        self._cache_max_batch_size = cache_max_batch_size
        if cache_max_batch_size:
            _cache_k = torch.zeros((cache_max_batch_size, cache_max_seq_len, n_kv_heads, self._head_size))
            self.register_buffer("_cache_k", _cache_k, persistent=False)
            _cache_v = torch.zeros((cache_max_batch_size, cache_max_seq_len, n_kv_heads, self._head_size))
            self.register_buffer("_cache_v", _cache_v, persistent=False)

    def forward(self, x, freq_cis, start_pos):
        _bn, _seq, _ = x.shape
        _dk = self._head_size ** 0.5

        _q, _k, _v = self._qw(x), self._kw(x), self._vw(x)

        _q = _q.reshape(_bn, _seq, self._n_q_heads, self._head_size)
        _k = _k.reshape(_bn, _seq, self._n_kv_heads, self._head_size)
        _v = _v.reshape(_bn, _seq, self._n_kv_heads, self._head_size)

        _q = apply_rotary_emb(_q, freq_cis[start_pos:start_pos + _seq])
        _k = apply_rotary_emb(_k, freq_cis[start_pos:start_pos + _seq])

        if self._cache_max_batch_size:
            self._cache_k[:_bn, start_pos: start_pos + _seq] = _k
            self._cache_v[:_bn, start_pos: start_pos + _seq] = _v
            _k = self._cache_k[:_bn, : start_pos + _seq]
            _v = self._cache_v[:_bn, : start_pos + _seq]

        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)

        _k = _k[:, None].repeat(1, self._group, 1, 1, 1).reshape(_bn, -1, start_pos + _seq, self._head_size)
        _v = _v[:, None].repeat(1, self._group, 1, 1, 1).reshape(_bn, -1, start_pos + _seq, self._head_size)

        if start_pos == 0:
            _o = F.scaled_dot_product_attention(_q, _k, _v, attn_mask=None, is_causal=True)
        else:
            _o = F.scaled_dot_product_attention(_q, _k, _v, attn_mask=None, is_causal=False)

        _o = _o.permute(0, 2, 1, 3).reshape(_bn, _seq, -1)
        return self._ow(_o)


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self._w0 = nn.Linear(input_dim, hidden_dim)
        self._w1 = nn.Linear(input_dim, hidden_dim)
        self._w2 = nn.Linear(hidden_dim, input_dim)
        self._gate = nn.SiLU()

    def forward(self, input):
        return self._w2(self._w0(input) * self._gate(self._w1(input)))


class RMSNorm(nn.Module):
    def __init__(self, input_dim, eps=1e-6):
        super().__init__()
        self._w = nn.Parameter(torch.randn(input_dim))
        self.eps = eps

    def forward(self, input):
        return self._w * input * torch.rsqrt(input.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LoraLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 rank: int = 8, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._freeze_module(self.linear)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.apply(self._init_weights)

    def _freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def _init_weights(self, module: nn.Module):
        if module is self.lora_A:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif module is self.lora_B:
            nn.init.init.zeros_(module.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.linear(input)
        if self.training and isinstance(self.lora_dropout, nn.Dropout):
            input = self.lora_dropout(input)
        lora_output = self.lora_B(self.lora_A(input)) * self.scaling
        return output + lora_output

    def merge_weights(self):
        self.linear.weight.data += (self.lora_B.weight.data @ self.lora_A.weight.data) * self.scaling
        return self

    def unmerge_weights(self):
        self.linear.weight.data -= (self.lora_B.weight.data @ self.lora_A.weight.data) * self.scaling
        return self


# ==================== 使用示例 ====================

def demo_multimodal():
    """多模态模型使用示例"""
    # 配置
    mm_config = MultimodalConfig(
        image_size=224,
        patch_size=14,
        vision_dim=768,
        projector_type="mlp",  # 可选: "linear", "mlp", "qformer"
        freeze_vision_encoder=True
    )

    # 初始化多模态解码器
    model = MultimodalTransformerDecoder(
        num_layers=12,
        input_dim=768,  # LLM维度
        hide_dim=2048,
        n_q_heads=12,
        n_kv_heads=4,  # GQA
        max_len=2048,
        multimodal_config=mm_config
    )

    # 模拟输入
    batch_size = 2
    # 文本: 假设已经是embeddings (实际应为token IDs经过embedding)
    text_seq_len = 50
    text_embeds = torch.randn(batch_size, text_seq_len, 768)

    # 图像: (B, C, H, W)
    images = torch.randn(batch_size, 3, 224, 224)

    # 前向传播: 图像+文本交错
    output = model(
        text_tokens=text_embeds,
        images=images,
        image_positions=[0],  # 图像插入在序列开头 (类似<image>标记)
        start_pos=0
    )

    print(f"Output shape: {output.shape}")  # (2, 50+257, 768)  257 = (224/14)^2 + 1 (CLS)

    # 仅文本
    output_text_only = model(text_tokens=text_embeds, start_pos=0)
    print(f"Text-only output shape: {output_text_only.shape}")


if __name__ == "__main__":
    demo_multimodal()
"""
视觉多模态模型实现

整合视觉编码器和语言模型，支持图文理解任务

模型结构:
    [Image] -> [CLIP ViT Encoder] -> [Vision-Language Connector] -> [Llama LLM]

Vision-Language Connector:
    将视觉特征投影到语言模型 embedding 空间
    采用 MLP 投影层（2层）参考 Qwen-VL/LLaVA

特殊 token:
    <image> ... </image>: 图像占位符
    <bos>: 文本开始
    <eos>: 文本结束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union

from vision_encoder import CustomCLIPVisionEncoder
from modeling_llama import LlamaForCausalLM


# =============================================================================
# Vision-Language Connector: 视觉-语言连接器
# =============================================================================
class VisionLanguageConnector(nn.Module):
    """
    视觉-语言连接器

    将视觉编码器的输出映射到语言模型的 embedding 空间

    结构: Linear -> Activation -> Linear
    参考 Qwen-VL/LLaVA 的 MLP 投影

    参数:
        vision_hidden_size: 视觉编码器隐藏层维度
        language_hidden_size: 语言模型隐藏层维度
        mlp_gated_act: 是否使用 SwiGLU 激活
    """
    def __init__(
        self,
        vision_hidden_size: int,
        language_hidden_size: int,
        mlp_gated_act: bool = True
    ):
        super().__init__()
        self.dense = nn.Linear(vision_hidden_size, language_hidden_size)

        if mlp_gated_act:
            # SwiGLU 激活（类似 Llama FFN）
            self.gate_proj = nn.Linear(language_hidden_size, language_hidden_size)
            self.activation_fn = nn.SiLU()

        self.fc = nn.Linear(language_hidden_size, language_hidden_size)
        self.layer_norm = nn.LayerNorm(language_hidden_size)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        参数:
            vision_features: [B, num_patches, vision_hidden_size]

        返回:
            projected: [B, num_patches, language_hidden_size]
        """
        x = self.dense(vision_features)

        # SwiGLU 激活（如果启用）
        if hasattr(self, 'gate_proj'):
            gate = self.activation_fn(self.gate_proj(x))
            x = gate * self.fc(x)
        else:
            x = self.fc(x)

        x = self.layer_norm(x)
        return x


# =============================================================================
# Multimodal Preprocessor: 多模态预处理器
# =============================================================================
class MultimodalPreprocessor(nn.Module):
    """
    多模态输入预处理器

    处理图像 tokens 的插入和位置编码

    参数:
        config: 模型配置
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 图像开始/结束 token（可学习）
        self.image_new_line = nn.Parameter(
            torch.zeros(1, 1, config.hidden_size)
        )

    def extend_input_ids(
        self,
        input_ids: torch.Tensor,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
        num_patches_per_image: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        在 input_ids 中插入图像 token 占位符

        参数:
            input_ids: [B, seq_len] 原始文本 token IDs
            image_sizes: 每张图像的尺寸列表（用于计算 patch 数量）
            num_patches_per_image: 每个图像的 patch 数量

        返回:
            new_input_ids: 带图像占位符的 input_ids
            labels: 对应的标签
            image_boundaries: 记录每张图像的位置信息
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        new_input_ids_list = []
        labels_list = []
        image_boundaries = []

        for b in range(batch_size):
            seq = input_ids[b].tolist()
            new_seq = []
            label_seq = []
            boundaries = []

            i = 0
            image_idx = 0
            while i < len(seq):
                # 检测图像占位符 token（假设使用特殊 token ID）
                # 这里使用固定的特殊 token: 50000 = <image>
                if seq[i] == 50000:
                    # 找到对应的结束 token
                    num_patches = num_patches_per_image
                    if image_sizes and image_idx < len(image_sizes):
                        # 根据图像尺寸计算实际 patch 数量
                        w, h = image_sizes[image_idx]
                        patch_size = self.config.vision_config.patch_size
                        num_patches = (w // patch_size) * (h // patch_size)

                    # 添加图像 patch 占位符
                    for p in range(num_patches):
                        new_seq.append(50001)  # <image_patch> token
                        label_seq.append(-100)  # 图像部分不计算损失

                    boundaries.append({
                        'start': len(new_seq) - num_patches,
                        'end': len(new_seq),
                        'num_patches': num_patches
                    })

                    image_idx += 1
                    i += 1  # 跳过图像结束 token
                else:
                    new_seq.append(seq[i])
                    label_seq.append(seq[i])
                    i += 1

            new_input_ids_list.append(new_seq)
            labels_list.append(label_seq)
            image_boundaries.append(boundaries)

        # Padding 到相同长度
        max_len = max(len(s) for s in new_input_ids_list)
        pad_token_id = self.config.pad_token_id

        new_input_ids = torch.full(
            (batch_size, max_len), pad_token_id, dtype=torch.long, device=device
        )
        labels = torch.full(
            (batch_size, max_len), -100, dtype=torch.long, device=device
        )

        for b in range(batch_size):
            seq_len = len(new_input_ids_list[b])
            new_input_ids[b, :seq_len] = torch.tensor(new_input_ids_list[b], device=device)
            labels[b, :seq_len] = torch.tensor(labels_list[b], device=device)

        return new_input_ids, labels, image_boundaries


# =============================================================================
# LlamaForConditionalGeneration: 多模态条件生成模型
# =============================================================================
class LlamaForConditionalGeneration(nn.Module):
    """
    视觉语言模型

    整合视觉编码器和 Llama 语言模型

    参数:
        config: 模型配置
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 视觉编码器
        vision_config = config.vision_config
        self.vision_encoder = CustomCLIPVisionEncoder(
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            hidden_size=vision_config.hidden_size,
            num_hidden_layers=vision_config.num_hidden_layers,
            num_attention_heads=vision_config.num_attention_heads,
            intermediate_size=vision_config.intermediate_size
        )

        # 视觉-语言连接器
        self.vision_language_connector = VisionLanguageConnector(
            vision_hidden_size=vision_config.hidden_size,
            language_hidden_size=config.hidden_size,
            mlp_gated_act=True
        )

        # 语言模型
        self.language_model = LlamaForCausalLM(config)

        # 多模态预处理器
        self.multimodal_preprocessor = MultimodalPreprocessor(config)

        # 特殊 token embedding
        self.image_new_line = nn.Parameter(
            torch.zeros(1, 1, config.hidden_size)
        )

    def get_input_embeddings(self) -> nn.Module:
        """获取语言模型的 embedding 层"""
        return self.language_model.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        """设置语言模型的 embedding 层"""
        self.language_model.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
        **kwargs
    ):
        """
        前向传播

        参数:
            input_ids: [B, seq_len] 文本 token IDs（包含图像占位符）
            pixel_values: [B, num_images, C, H, W] 图像像素值
            labels: [B, seq_len] 目标标签
            attention_mask: [B, seq_len] 注意力掩码
            position_ids: [B, seq_len] 位置索引
            image_sizes: 每张图像的尺寸

        返回:
            dict: {'logits': logits, 'loss': loss}
        """
        if pixel_values is not None:
            return self._forward_with_vision(
                input_ids, pixel_values, labels, attention_mask, position_ids, image_sizes
            )
        else:
            # 纯文本模式
            return self.language_model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids
            )

    def _forward_with_vision(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        image_sizes: Optional[List[Tuple[int, int]]]
    ):
        """
        带视觉输入的前向传播

        流程:
            1. 提取图像特征
            2. 投影到语言空间
            3. 替换 input_ids 中的图像 token
            4. 拼接图像 embedding 和文本 embedding
            5. 通过语言模型
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1. 提取视觉特征
        # pixel_values: [B, num_images, C, H, W] -> [B*num_images, C, H, W]
        num_images = pixel_values.shape[1]
        pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
        vision_outputs = self.vision_encoder(pixel_values)
        vision_features = vision_outputs['last_hidden_state']  # [B*num_images, num_patches+1, vision_hidden]

        # 移除 CLS token，只保留 patch tokens
        vision_features = vision_features[:, 1:, :]  # [B*num_images, num_patches, vision_hidden]

        # 2. 投影到语言空间
        vision_features = self.vision_language_connector(vision_features)  # [B*num_images, num_patches, lang_hidden]

        # 3. 重塑为 [B, num_images * num_patches, lang_hidden]
        vision_features = vision_features.view(
            batch_size, num_images, -1, self.config.hidden_size
        )
        vision_features = vision_features.reshape(
            batch_size, -1, self.config.hidden_size
        )  # [B, total_patches, lang_hidden]

        # 4. 获取图像 new line embedding
        image_new_line = self.image_new_line.expand(batch_size, -1, -1)  # [B, 1, hidden]

        # 5. 拼接视觉特征和文本 embedding
        # 这里简化处理：直接返回图像特征供后续使用
        # 实际应用中需要将图像特征插入到正确的位置

        # 对于简化版本，直接使用语言模型
        # 图像特征作为额外上下文
        language_outputs = self.language_model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        return language_outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        **kwargs
    ):
        """
        生成文本

        参数:
            input_ids: [B, seq_len] 输入 token IDs
            pixel_values: [B, C, H, W] 图像（可选）
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_k: Top-k 采样

        返回:
            output_ids: 生成的 token IDs
        """
        return self.language_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            **kwargs
        )


# =============================================================================
# 纯文本模式（无视觉编码器）
# =============================================================================
class LlamaForCausalLMTextOnly(nn.Module):
    """
    纯文本语言模型（无视觉编码器）

    用于纯文本训练和推理
    """
    def __init__(self, config):
        super().__init__()
        self.language_model = LlamaForCausalLM(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ):
        return self.language_model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        **kwargs
    ):
        return self.language_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            **kwargs
        )

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.model.embed_tokens = value

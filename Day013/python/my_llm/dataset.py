"""
数据集模块

生成随机训练数据用于测试模型训练流程
支持：
1. 随机文本数据生成
2. 合成图像生成
3. 多模态数据打包
"""

import os
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import torch
from torch.utils.data import Dataset

# 导入 tokenizer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'upload_huggingface'))
try:
    from tokenizer import SentencePieceTokenizer
except ImportError:
    # 如果导入失败，创建一个简单的 tokenizer
    SentencePieceTokenizer = None


# =============================================================================
# 合成图像生成
# =============================================================================
def generate_synthetic_image(
    width: int = 224,
    height: int = 224,
    pattern: str = "random"
) -> Image.Image:
    """
    生成合成图像

    参数:
        width: 图像宽度
        height: 图像高度
        pattern: 图案类型 ("random", "gradient", "checkerboard", "noise")

    返回:
        PIL.Image 对象
    """
    if pattern == "random":
        # 随机噪声图像
        data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    elif pattern == "gradient":
        # 渐变图像
        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)
        xx, yy = np.meshgrid(x, y)
        data = np.stack([xx, yy, (xx + yy) // 2], axis=-1)
    elif pattern == "checkerboard":
        # 棋盘格图像
        block_size = 16
        data = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    data[i:i+block_size, j:j+block_size] = [255, 255, 255]
                else:
                    data[i:i+block_size, j:j+block_size] = [0, 0, 0]
    elif pattern == "noise":
        # 彩色噪声
        data = np.random.normal(128, 50, (height, width, 3))
        data = np.clip(data, 0, 255).astype(np.uint8)
    else:
        data = np.zeros((height, width, 3), dtype=np.uint8)

    return Image.fromarray(data)


def save_synthetic_images(
    output_dir: str,
    num_images: int = 100,
    image_size: Tuple[int, int] = (224, 224),
    patterns: List[str] = None
) -> List[str]:
    """
    生成并保存合成图像

    参数:
        output_dir: 输出目录
        num_images: 生成图像数量
        image_size: 图像尺寸 (width, height)
        patterns: 图案类型列表

    返回:
        图像路径列表
    """
    os.makedirs(output_dir, exist_ok=True)

    if patterns is None:
        patterns = ["random", "gradient", "checkerboard", "noise"]

    image_paths = []
    for i in range(num_images):
        pattern = random.choice(patterns)
        img = generate_synthetic_image(
            width=image_size[0],
            height=image_size[1],
            pattern=pattern
        )

        # 添加一些随机变换
        if random.random() > 0.5:
            # 随机旋转
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=(128, 128, 128))

        if random.random() > 0.5:
            # 随机添加边框
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            border_color = tuple(random.randint(0, 255) for _ in range(3))
            draw.rectangle([0, 0, img.width-1, img.height-1], outline=border_color, width=3)

        # 保存
        img_path = os.path.join(output_dir, f"synthetic_{i:04d}.jpg")
        img.save(img_path, quality=95)
        image_paths.append(img_path)

    return image_paths


# =============================================================================
# 文本数据生成
# =============================================================================
def generate_random_text(
    min_length: int = 10,
    max_length: int = 100,
    vocab_size: int = 50000
) -> List[int]:
    """
    生成随机 token 序列（模拟文本）

    参数:
        min_length: 最小序列长度
        max_length: 最大序列长度
        vocab_size: 词汇表大小

    返回:
        token IDs 列表
    """
    length = random.randint(min_length, max_length)
    tokens = [random.randint(4, vocab_size - 1) for _ in range(length)]
    return tokens


def generate_text_dataset(
    num_samples: int = 1000,
    min_length: int = 8,
    max_length: int = 64,
    vocab_size: int = 50000
) -> List[Dict]:
    """
    生成文本数据集

    参数:
        num_samples: 样本数量
        min_length: 最小序列长度
        max_length: 最大序列长度
        vocab_size: 词汇表大小

    返回:
        数据列表
    """
    data = []
    for i in range(num_samples):
        # 输入序列
        input_ids = generate_random_text(min_length, max_length, vocab_size)

        # 目标序列（输入右移一位）
        labels = input_ids[1:] + [random.randint(4, vocab_size - 1)]

        data.append({
            'id': i,
            'input_ids': input_ids,
            'labels': labels
        })

    return data


# =============================================================================
# 多模态数据生成
# =============================================================================
def generate_multimodal_data(
    num_samples: int = 100,
    text_min_length: int = 8,
    text_max_length: int = 32,
    vocab_size: int = 50000,
    image_dir: Optional[str] = None,
    image_paths: Optional[List[str]] = None
) -> List[Dict]:
    """
    生成多模态数据集

    参数:
        num_samples: 样本数量
        text_min_length: 文本最小长度
        text_max_length: 文本最大长度
        vocab_size: 词汇表大小
        image_dir: 图像目录
        image_paths: 图像路径列表

    返回:
        多模态数据列表
    """
    data = []

    # 如果没有提供图像路径，生成合成图像
    if image_paths is None:
        if image_dir is None:
            image_dir = os.path.join(os.path.dirname(__file__), "data", "images")

        # 生成合成图像
        image_paths = save_synthetic_images(image_dir, num_samples // 2 + 1)

    for i in range(num_samples):
        # 文本部分
        has_image = random.random() > 0.5  # 50% 概率包含图像

        # 添加图像占位符 token
        text_tokens = [50000]  # <image> token

        if has_image and image_paths:
            # 添加一些文本
            prefix = generate_random_text(text_min_length // 2, text_max_length // 2, vocab_size)
            suffix = generate_random_text(text_min_length // 2, text_max_length // 2, vocab_size)
            text_tokens = prefix + text_tokens + suffix
            image_path = random.choice(image_paths)
        else:
            text_tokens = generate_random_text(text_min_length, text_max_length, vocab_size)
            image_path = None

        # 标签（图像部分为 -100，不计算损失）
        labels = [-100 if t == 50000 else t for t in text_tokens]

        sample = {
            'id': i,
            'input_ids': text_tokens,
            'labels': labels,
            'has_image': has_image,
            'image_path': image_path
        }

        data.append(sample)

    return data


# =============================================================================
# PyTorch Dataset
# =============================================================================
class RandomTextDataset(Dataset):
    """
    随机文本数据集

    用于测试训练流程
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 64,
        vocab_size: int = 50000,
        pad_token_id: int = 0
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        # 预生成所有数据
        self.data = generate_text_dataset(
            num_samples=num_samples,
            min_length=seq_length // 2,
            max_length=seq_length,
            vocab_size=vocab_size
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        input_ids = sample['input_ids']
        labels = sample['labels']

        # Padding 到固定长度
        if len(input_ids) < self.seq_length:
            pad_len = self.seq_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [self.pad_token_id] * pad_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class MultimodalDataset(Dataset):
    """
    多模态数据集

    支持文本和图像
    """

    def __init__(
        self,
        data: List[Dict],
        seq_length: int = 128,
        vocab_size: int = 50000,
        pad_token_id: int = 0,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.data = data
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        input_ids = sample['input_ids']
        labels = sample['labels']

        # Padding
        if len(input_ids) < self.seq_length:
            pad_len = self.seq_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [self.pad_token_id] * pad_len

        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

        # 如果有图像，加载图像
        if sample.get('image_path') and os.path.exists(sample['image_path']):
            try:
                img = Image.open(sample['image_path']).convert('RGB')
                img = img.resize(self.image_size)

                # 转换为 tensor
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [C, H, W]

                # ImageNet 归一化
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std

                result['pixel_values'] = img_tensor
                result['image_sizes'] = list(self.image_size)
            except Exception as e:
                # 如果图像加载失败，返回零图像
                result['pixel_values'] = torch.zeros(3, *self.image_size)
                result['image_sizes'] = list(self.image_size)
        else:
            result['pixel_values'] = torch.zeros(3, *self.image_size)
            result['image_sizes'] = list(self.image_size)

        return result


class HuggingFaceDataset(Dataset):
    """
    HuggingFace 风格数据集

    从 JSON 文件加载数据
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
        image_dir: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_dir = image_dir

        # 加载数据
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Tokenize 文本
        text = item.get('text', '')
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)
        }

        # 处理标签中的 padding
        result['labels'][result['labels'] == self.tokenizer.pad_token_id] = -100

        # 处理图像（如果有）
        if 'image_path' in item and self.image_dir:
            image_path = os.path.join(self.image_dir, item['image_path'])
            if os.path.exists(image_path):
                img = Image.open(image_path).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
                result['pixel_values'] = img_tensor

        return result


# =============================================================================
# 数据集工厂
# =============================================================================
class DatasetFactory:
    """数据集工厂"""

    @staticmethod
    def create_text_dataset(
        num_samples: int = 1000,
        seq_length: int = 64,
        vocab_size: int = 50000
    ) -> RandomTextDataset:
        """创建随机文本数据集"""
        return RandomTextDataset(
            num_samples=num_samples,
            seq_length=seq_length,
            vocab_size=vocab_size
        )

    @staticmethod
    def create_multimodal_dataset(
        num_samples: int = 100,
        seq_length: int = 128,
        vocab_size: int = 50000,
        image_dir: Optional[str] = None
    ) -> MultimodalDataset:
        """创建多模态数据集"""
        # 生成合成数据
        data = generate_multimodal_data(
            num_samples=num_samples,
            image_dir=image_dir
        )

        return MultimodalDataset(
            data=data,
            seq_length=seq_length,
            vocab_size=vocab_size
        )

    @staticmethod
    def create_hf_dataset(
        data_path: str,
        tokenizer,
        max_length: int = 128,
        image_dir: Optional[str] = None
    ) -> HuggingFaceDataset:
        """创建 HuggingFace 风格数据集"""
        return HuggingFaceDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            image_dir=image_dir
        )

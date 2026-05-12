"""
测试脚本

验证模型输出格式是否正确
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(__file__))

from config import ModelConfig, VisionConfig, ConfigFactory
from modeling_llama import LlamaForCausalLM, LlamaModel, RMSNorm, LlamaAttention, LlamaMLP
from vision_encoder import CustomCLIPVisionEncoder
from modeling_vlm import LlamaForConditionalGeneration


def test_rmsnorm():
    """测试 RMSNorm"""
    print("\n" + "="*50)
    print("Testing RMSNorm...")
    print("="*50)

    batch_size = 2
    seq_len = 10
    hidden_size = 128

    norm = RMSNorm(hidden_size)
    x = torch.randn(batch_size, seq_len, hidden_size)
    y = norm(x)

    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print("  RMSNorm test passed!")


def test_rope():
    """测试 RoPE"""
    print("\n" + "="*50)
    print("Testing RoPE...")
    print("="*50)

    from modeling_llama import precompute_freqs_cis, apply_rotary_pos_emb

    batch_size = 2
    seq_len = 10
    num_heads = 4
    head_dim = 32
    max_len = 128

    # 预计算频率
    freqs_cis = precompute_freqs_cis(head_dim, max_len)
    print(f"  Freqs cis shape: {freqs_cis.shape}")

    # 创建 Q, K
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # 应用 RoPE
    q_rot, k_rot = apply_rotary_pos_emb(q, k, freqs_cis)

    assert q_rot.shape == q.shape, f"Q shape mismatch: {q_rot.shape} vs {q.shape}"
    assert k_rot.shape == k.shape, f"K shape mismatch: {k_rot.shape} vs {k.shape}"
    print(f"  Q shape: {q.shape} -> {q_rot.shape}")
    print(f"  K shape: {k.shape} -> {k_rot.shape}")
    print("  RoPE test passed!")


def test_attention():
    """测试注意力机制"""
    print("\n" + "="*50)
    print("Testing Attention...")
    print("="*50)

    from modeling_llama import precompute_freqs_cis

    batch_size = 2
    seq_len = 16
    hidden_size = 128
    num_heads = 4
    num_kv_heads = 4  # 保持一致，测试 GQA 需要单独处理
    head_dim = 32
    max_len = 64

    attn = LlamaAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim
    )

    x = torch.randn(batch_size, seq_len, hidden_size)
    freqs_cis = precompute_freqs_cis(head_dim, max_len)

    y = attn(x, freqs_cis)

    assert y.shape == (batch_size, seq_len, hidden_size), \
        f"Shape mismatch: {y.shape} vs {(batch_size, seq_len, hidden_size)}"
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print("  Attention test passed!")


def test_mlp():
    """测试 MLP"""
    print("\n" + "="*50)
    print("Testing SwiGLU MLP...")
    print("="*50)

    batch_size = 2
    seq_len = 16
    hidden_size = 128
    intermediate_size = 256

    mlp = LlamaMLP(hidden_size, intermediate_size)
    x = torch.randn(batch_size, seq_len, hidden_size)
    y = mlp(x)

    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print("  SwiGLU MLP test passed!")


def test_llama_model():
    """测试完整 Llama 模型"""
    print("\n" + "="*50)
    print("Testing LlamaModel...")
    print("="*50)

    config = ModelConfig(
        vocab_size=4096,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128
    )

    model = LlamaModel(config)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    hidden_states = model(input_ids)

    assert hidden_states.shape == (batch_size, seq_len, config.hidden_size), \
        f"Shape mismatch: {hidden_states.shape}"
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Output shape: {hidden_states.shape}")
    print("  LlamaModel test passed!")


def test_llama_for_causal_lm():
    """测试因果语言模型"""
    print("\n" + "="*50)
    print("Testing LlamaForCausalLM...")
    print("="*50)

    config = ModelConfig(
        vocab_size=4096,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128,
        share_input_output_embeddings=False
    )

    model = LlamaForCausalLM(config)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(4, config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids, labels=labels)

    assert 'logits' in outputs, "Missing 'logits' in outputs"
    assert 'loss' in outputs, "Missing 'loss' in outputs"
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size), \
        f"Logits shape mismatch: {outputs['logits'].shape}"
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print("  LlamaForCausalLM test passed!")


def test_vision_encoder():
    """测试视觉编码器"""
    print("\n" + "="*50)
    print("Testing CustomCLIPVisionEncoder...")
    print("="*50)

    encoder = CustomCLIPVisionEncoder(
        image_size=224,
        patch_size=14,
        hidden_size=768,
        num_hidden_layers=4,  # 减少层数加快测试
        num_attention_heads=4,
        intermediate_size=1024
    )
    print(f"  Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 224, 224)

    outputs = encoder(pixel_values)

    assert 'last_hidden_state' in outputs, "Missing 'last_hidden_state'"
    assert 'pooler_output' in outputs, "Missing 'pooler_output'"

    num_patches = (224 // 14) ** 2 + 1  # 256 + 1 CLS
    expected_shape = (batch_size, num_patches, 768)
    assert outputs['last_hidden_state'].shape == expected_shape, \
        f"Shape mismatch: {outputs['last_hidden_state'].shape} vs {expected_shape}"

    assert outputs['pooler_output'].shape == (batch_size, 768), \
        f"Pooler output shape mismatch: {outputs['pooler_output'].shape}"

    print(f"  Input shape: {pixel_values.shape}")
    print(f"  Last hidden state shape: {outputs['last_hidden_state'].shape}")
    print(f"  Pooler output shape: {outputs['pooler_output'].shape}")
    print("  Vision Encoder test passed!")


def test_generation():
    """测试生成"""
    print("\n" + "="*50)
    print("Testing Generation...")
    print("="*50)

    config = ModelConfig(
        vocab_size=4096,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128
    )

    model = LlamaForCausalLM(config)
    model.eval()

    batch_size = 1
    seq_len = 8
    input_ids = torch.randint(4, config.vocab_size, (batch_size, seq_len))

    output_ids = model.generate(
        input_ids,
        max_new_tokens=16,
        temperature=0.7,
        top_k=20
    )

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {output_ids.shape}")
    print(f"  Input tokens: {input_ids[0].tolist()}")
    print(f"  Output tokens: {output_ids[0].tolist()}")
    print("  Generation test passed!")


def test_training_step():
    """测试训练步骤"""
    print("\n" + "="*50)
    print("Testing Training Step...")
    print("="*50)

    config = ModelConfig(
        vocab_size=4096,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128
    )

    model = LlamaForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(4, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(4, config.vocab_size, (batch_size, seq_len))

    # 前向传播
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']

    # 反向传播
    loss.backward()

    # 更新
    optimizer.step()
    optimizer.zero_grad()

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed: {all(p.grad is not None for p in model.parameters() if p.requires_grad)}")
    print("  Training step test passed!")


def test_save_and_load():
    """测试保存和加载"""
    print("\n" + "="*50)
    print("Testing Save and Load...")
    print("="*50)

    config = ModelConfig(
        vocab_size=4096,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128
    )

    # 创建模型
    model1 = LlamaForCausalLM(config)

    # 保存
    save_path = "/tmp/test_model.pt"
    torch.save({
        'model_state_dict': model1.state_dict(),
        'config': config.to_dict()
    }, save_path)
    print(f"  Model saved to: {save_path}")

    # 加载
    checkpoint = torch.load(save_path, map_location='cpu')
    model2 = LlamaForCausalLM(ModelConfig.from_dict(checkpoint['config']))
    model2.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Model loaded from: {save_path}")

    # 验证
    batch_size = 1
    seq_len = 8
    input_ids = torch.randint(4, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        out1 = model1(input_ids)
        out2 = model2(input_ids)

    assert torch.allclose(out1['logits'], out2['logits']), "Outputs don't match!"
    print("  Save and load test passed!")


def test_output_format():
    """测试输出格式（与 HuggingFace 一致性）"""
    print("\n" + "="*50)
    print("Testing Output Format (HuggingFace compatibility)...")
    print("="*50)

    config = ModelConfig(
        vocab_size=4096,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128
    )

    model = LlamaForCausalLM(config)
    model.eval()

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(4, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(4, config.vocab_size, (batch_size, seq_len))

    # HF 格式应该是字典
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    # 检查输出格式
    print("  Output keys:", list(outputs.keys()))

    assert isinstance(outputs, dict), "Output should be a dict"
    assert 'logits' in outputs, "Missing 'logits'"
    assert outputs['logits'].dtype == torch.float32, f"Wrong dtype: {outputs['logits'].dtype}"

    # 检查 logits 形状
    expected_logits_shape = (batch_size, seq_len, config.vocab_size)
    assert outputs['logits'].shape == expected_logits_shape, \
        f"Wrong logits shape: {outputs['logits'].shape} vs {expected_logits_shape}"

    # 检查 loss
    if outputs['loss'] is not None:
        print(f"  Loss value: {outputs['loss'].item():.4f}")
        assert outputs['loss'].ndim == 0, f"Loss should be scalar, got shape {outputs['loss'].shape}"

    print("  Logits shape:", outputs['logits'].shape)
    print("  Logits dtype:", outputs['logits'].dtype)
    print("  Output format test passed!")


def main():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# Model Tests")
    print("#"*60)

    # 测试基础组件
    test_rmsnorm()
    test_rope()

    # 测试模型组件
    test_attention()
    test_mlp()

    # 测试完整模型
    test_llama_model()
    test_llama_for_causal_lm()

    # 测试视觉编码器
    test_vision_encoder()

    # 测试生成
    test_generation()

    # 测试训练
    test_training_step()

    # 测试保存加载
    test_save_and_load()

    # 测试输出格式
    test_output_format()

    print("\n" + "#"*60)
    print("# All Tests Passed!")
    print("#"*60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
将 DeepSpeed 训练的 pt 模型转换为 Ollama 可用的格式
Ollama 支持直接使用 HuggingFace 格式的模型
"""
import os
import sys
import json
import torch
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/data/Workspace/airelearn/Day011/python/my_llm')

# 从 upload_huggingface 目录复制 model.py 以获取 StorierModel 和 StorierConfig
# 这里直接内联定义，因为需要确保路径正确
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache


def precompute_freqs_cis(dim, end, theta=50000.0, device="cpu"):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, freqs_cis):
    assert xq.shape[-1] % 2 == 0, "The last dimension of xq must be even."
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[1]][None, :, None]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)


class RMSNorm(nn.Module):
    def __init__(self, input_dim, eps=1e-6):
        super().__init__()
        self._w = nn.Parameter(torch.randn(input_dim))
        self.eps = eps

    def forward(self, input):
        return self._w * input * torch.rsqrt(input.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class TransformerLayer(nn.Module):
    def __init__(self, input_dim, hide_dim, n_q_heads, n_kv_heads):
        super().__init__()
        self._att_norm = RMSNorm(input_dim)
        self._att_layer = MultiHeadAttention(input_dim, hide_dim, n_q_heads, n_kv_heads)
        self._ffn_norm = RMSNorm(input_dim)
        self._ffn_layer = FFN(input_dim, hide_dim)

    def forward(self, x, freq_cis, start_pos, past_key_value=None, use_cache=False):
        _x = self._att_norm(x)
        attn_out, present = self._att_layer(_x, freq_cis, start_pos, past_key_value, use_cache)
        _x = x + attn_out
        _y = self._ffn_norm(_x)
        _y = _y + self._ffn_layer(_y)
        return _y, present


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hide_dim, n_q_heads, n_kv_heads):
        super().__init__()
        self._n_q_heads = n_q_heads
        self._n_kv_heads = n_kv_heads
        self._group = n_q_heads // n_kv_heads
        self._head_size = input_dim // self._n_q_heads
        self._qw = nn.Linear(input_dim, self._head_size * self._n_q_heads)
        self._kw = nn.Linear(input_dim, self._head_size * self._n_kv_heads)
        self._vw = nn.Linear(input_dim, self._head_size * self._n_kv_heads)
        self._ow = nn.Linear(self._head_size * self._n_q_heads, input_dim)

    def forward(self, x, freq_cis, start_pos, past_key_value=None, use_cache=False):
        from torch.nn import functional as F
        _bn, _seq, _ = x.shape
        _q, _k, _v = self._qw(x), self._kw(x), self._vw(x)
        _q = _q.reshape(_bn, _seq, self._n_q_heads, self._head_size)
        _k = _k.reshape(_bn, _seq, self._n_kv_heads, self._head_size)
        _v = _v.reshape(_bn, _seq, self._n_kv_heads, self._head_size)
        _q = apply_rotary_emb(_q, freq_cis[start_pos:start_pos + _seq])
        _k = apply_rotary_emb(_k, freq_cis[start_pos:start_pos + _seq])
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k is not None and past_v is not None:
                _k = torch.cat([past_k, _k], dim=1)
                _v = torch.cat([past_v, _v], dim=1)
        present_key_value = None
        if use_cache:
            present_key_value = (_k, _v)
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
        return self._ow(_o), present_key_value


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self._w0 = nn.Linear(input_dim, hidden_dim)
        self._w1 = nn.Linear(input_dim, hidden_dim)
        self._w2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, input):
        return self._w2(self._w0(input) * torch.nn.functional.silu(self._w1(input)))


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, input_dim, hide_dim, n_q_heads, n_kv_heads, max_len):
        super().__init__()
        self._layers = nn.ModuleList([
            TransformerLayer(input_dim, hide_dim, n_q_heads, n_kv_heads)
            for _ in range(num_layers)
        ])
        self._out_norm = RMSNorm(input_dim)
        _freq_cis = precompute_freqs_cis(input_dim // n_q_heads, max_len)
        self.register_buffer("freq_cis", _freq_cis, persistent=False)

    def forward(self, x, start_pos, past_key_values=None, use_cache=False):
        present_key_values = []
        if past_key_values is not None:
            if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
                past_kv_list = [(past_key_values.key_cache[i], past_key_values.value_cache[i])
                                for i in range(len(past_key_values.key_cache))]
            else:
                past_kv_list = past_key_values
        else:
            past_kv_list = None
        for i, layer in enumerate(self._layers):
            layer_past = past_kv_list[i] if past_kv_list is not None else None
            x, present = layer(x, self.freq_cis, start_pos, layer_past, use_cache)
            present_key_values.append(present)
        x = self._out_norm(x)
        return x, present_key_values


class StorierConfig(PretrainedConfig):
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
        self.pad_token_id = 0
        self.bos_token_id = 2
        self.eos_token_id = 3

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

    @property
    def num_hidden_layers(self):
        return self.num_layers


class StorierModel(PreTrainedModel, GenerationMixin):
    config_class = StorierConfig
    supports_gradient_checkpointing = True
    base_model_prefix = "storier"

    def __init__(self, config):
        super().__init__(config)
        self.emb = nn.Embedding(config.vocab_size, config.input_dim)
        self.tf_layer = TransformerDecoder(
            num_layers=config.num_layers,
            input_dim=config.input_dim,
            hide_dim=config.hide_dim,
            n_q_heads=config.n_q_heads,
            n_kv_heads=config.n_kv_heads,
            max_len=config.max_pos_len
        )
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=None, **kwargs):
        from torch.nn import functional as F
        if use_cache is None:
            use_cache = self.config.use_cache if hasattr(self.config, 'use_cache') else True

        if past_key_values is not None:
            if hasattr(past_key_values, 'key_cache'):
                key_cache, value_cache = past_key_values.to_legacy_cache()
                past_kv_list = []
                for i in range(len(key_cache)):
                    if key_cache[i] is not None and value_cache[i] is not None:
                        past_kv_list.append((key_cache[i], value_cache[i]))
                    else:
                        past_kv_list.append(None)
            else:
                past_kv_list = past_key_values
        else:
            past_kv_list = None

        start_pos = 0
        if past_key_values is not None:
            if hasattr(past_key_values, 'get_seq_length'):
                start_pos = past_key_values.get_seq_length()
            elif isinstance(past_key_values, list) and past_key_values[0] is not None:
                start_pos = past_key_values[0][0].shape[1]

        tokens = self.emb(input_ids)
        features, present_key_values = self.tf_layer(tokens, start_pos, past_kv_list, use_cache)
        logits = features @ self.emb.weight.T
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=present_key_values,
            hidden_states=None, attentions=None,
        )

    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, value):
        self.emb = value


def convert_pt_to_huggingface():
    """将 DeepSpeed pt 权重转换为 HuggingFace 格式"""
    cache_dir = Path("/data/Workspace/airelearn/Day011/python/my_llm/cache")
    weights_path = Path("/data/Workspace/airelearn/Day011/python/my_llm/weights/9/mp_rank_00_model_states.pt")

    print(f"加载权重: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["module"]
    print(f"原始权重键数量: {len(state_dict)}")

    # 过滤掉非模型权重
    model_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_")}
    print(f"模型权重键数量: {len(model_state_dict)}")

    # 加载配置
    config = StorierConfig(
        num_layers=48,
        input_dim=768,
        hide_dim=3072,
        n_q_heads=12,
        n_kv_heads=2,
        max_pos_len=16384,
        vocab_size=50000,
        use_cache=True
    )

    # 创建模型
    model = StorierModel(config)

    # 加载权重
    model.load_state_dict(model_state_dict, strict=False)
    print("权重加载完成")

    # 保存 HuggingFace 格式
    model.save_pretrained(cache_dir)
    print(f"模型保存到: {cache_dir}")

    # 保存配置文件（带 auto_map）
    config_dict = config.to_dict()
    config_dict["auto_map"] = {
        "AutoConfig": "model.StorierConfig",
        "AutoModelForCausalLM": "model.StorierForCausalLM",
    }

    config_file = cache_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    print(f"配置文件保存到: {config_file}")

    # 创建简单的 tokenizer（因为你的模型没有单独的 tokenizer）
    # Ollama 需要 tokenizer 文件
    create_tokenizer(cache_dir)

    return True


def create_tokenizer(cache_dir):
    """创建简单的 tokenizer 文件"""
    # 使用 transformers 的默认 tokenizer 作为基础
    # 或者创建一个自定义的
    from transformers import PreTrainedTokenizer, AddedToken

    class SimpleTokenizer(PreTrainedTokenizer):
        """简单的字符级 tokenizer"""
        vocab = {chr(i + ord('A')): i + 1 for i in range(26)}  # A-Z: 1-26
        vocab['<pad>'] = 0
        vocab['<bos>'] = 2
        vocab['<eos>'] = 3
        vocab['<unk>'] = 4

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._vocab = self.vocab.copy()
            self._id_to_token = {v: k for k, v in self._vocab.items()}

        @property
        def vocab_size(self):
            return len(self._vocab)

        def _tokenize(self, text, **kwargs):
            return list(text.upper())

        def _convert_token_to_id(self, token):
            return self._vocab.get(token, self._vocab['<unk>'])

        def _convert_id_to_token(self, index):
            return self._id_to_token.get(index, '<unk>')

        def get_vocab(self):
            return self._vocab.copy()

        def save_vocabulary(self, save_directory, filename_prefix=None):
            import json
            vocab_file = Path(save_directory) / "tokenizer.json"
            with open(vocab_file, 'w') as f:
                json.dump({"vocab": self._vocab}, f)
            return (str(vocab_file),)

        @property
        def pad_token_id(self):
            return 0

        @property
        def bos_token_id(self):
            return 2

        @property
        def eos_token_id(self):
            return 3

        @property
        def unk_token_id(self):
            return 4

    tokenizer = SimpleTokenizer()

    # 保存 tokenizer
    tokenizer.save_pretrained(cache_dir)
    print(f"Tokenizer 保存到: {cache_dir}")

    # 创建 tokenizer_config.json
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": True,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    with open(cache_dir / "tokenizer_config.json", 'w') as f:
        json.dump(tokenizer_config, f, indent=2)

    return tokenizer


if __name__ == "__main__":
    print("=" * 50)
    print("开始转换 DeepSpeed 模型到 Ollama 格式")
    print("=" * 50)
    success = convert_pt_to_huggingface()
    if success:
        print("\n转换成功!")
    else:
        print("\n转换失败!")
        sys.exit(1)
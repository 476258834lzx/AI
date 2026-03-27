import os
import json
import torch
import shutil
from torch import nn
# from Day011.python.transformer.diy_rope import *
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, GenerationConfig,GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import modeling_auto
from transformers.cache_utils import DynamicCache

def precompute_freqs_cis(dim,end,theta=50000.0,device="cpu"):#生成位置向量列表,设置单位角度
    freqs = 1.0/(theta ** (torch.arange(0,dim,2,device=device)[:(dim//2)].float()/dim))#对50000按步长开方的倒数,每个向量两两一组(dim=32,32/2=16),公式中的dim可以不分两组不取步长2,dim改为16与查询向量形状匹配
    t = torch.arange(end,device=device)#对应token数创建对应数量的旋转编码
    freqs = torch.outer(t,freqs).float()#求外积，20个token,16组向量，按不同token的下标t编码角度成倍偏转
    freqs_cis=torch.polar(torch.ones_like(freqs),freqs)#创建模为1的极坐标,横轴为实数1,纵轴为虚部1i,第一个参数为模长(张量),第二个参数为δ(张量),转极坐标后为cosδ+sinδi
    return freqs_cis

def apply_rotary_emb(xq,freqs_cis):#将词向量叠加位置向量,点乘,向量两两转到复平面
    assert xq.shape[-1] % 2 == 0, "The last dimension of xq must be even."
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1],-1,2))#查询向量分成两组
    freqs_cis = reshape_for_broadcast(freqs_cis,xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)#在复数域计算旋转，方便词向量与词向量做余弦，位置与位置做余弦,位置向量和embedding点乘按多项式计算法则,z1*z2=(ac-bd)+(ad+bc)j,(cosδcosβ-sinδsinβ)满足和角公式,结果为模长相乘,角度相加
    return xq_out.type_as(xq)

def reshape_for_broadcast(freqs_cis,x):#形状匹配#写了个bug,但是广播导致训练不受影响
    freqs_cises = freqs_cis[:x.shape[1]]
    return freqs_cises[None, :, None]

class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,  # 解码器的层数
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 max_len
                 ):
        super().__init__()

        self._layers = nn.ModuleList(
            [TransformerLayer(input_dim,
                              hide_dim,
                              n_q_heads,
                              n_kv_heads) for _ in range(num_layers)]
        )

        self._out_norm = RMSNorm(input_dim)

        _freq_cis = precompute_freqs_cis(input_dim // n_q_heads, max_len)

        self.register_buffer("freq_cis", _freq_cis, persistent=False)

    def forward(self, x, start_pos, past_key_values=None, use_cache=False):
        present_key_values = []
        # 转换 past_key_values 为兼容格式
        if past_key_values is not None:
            if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
                # 如果是 DynamicCache，提取 (k, v) 列表
                past_kv_list = [(past_key_values.key_cache[i], past_key_values.value_cache[i])
                                for i in range(len(past_key_values.key_cache))]
            else:
                # 假设已经是列表格式
                past_kv_list = past_key_values
        else:
            past_kv_list = None

        for i, layer in enumerate(self._layers):
            layer_past = past_kv_list[i] if past_kv_list is not None else None
            x, present = layer(x, self.freq_cis, start_pos, layer_past, use_cache)
            present_key_values.append(present)
        x = self._out_norm(x)

        return x, present_key_values


class TransformerLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads):
        super().__init__()

        self._att_norm = RMSNorm(input_dim)

        self._att_layer = MultiHeadAttention(input_dim,
                                            hide_dim,
                                            n_q_heads,
                                            n_kv_heads)

        self._ffn_norm = RMSNorm(input_dim)

        self._ffn_layer = FFN(input_dim,hide_dim)

    def forward(self, x, freq_cis, start_pos, past_key_value=None, use_cache=False):
        _x = x
        _x = self._att_norm(_x)
        attn_out, present_key_value = self._att_layer(_x, freq_cis, start_pos, past_key_value, use_cache)

        _x = x + attn_out

        _y = _x
        _y = self._ffn_norm(_y)
        _y = self._ffn_layer(_y)

        _y = _y + _x

        return _y, present_key_value


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,

                 ):
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
        _bn, _seq, _ = x.shape
        _dk = self._head_size ** 0.5

        _q, _k, _v = self._qw(x), self._kw(x), self._vw(x)

        _q = _q.reshape(_bn, _seq, self._n_q_heads, self._head_size)
        _k = _k.reshape(_bn, _seq, self._n_kv_heads, self._head_size)
        _v = _v.reshape(_bn, _seq, self._n_kv_heads, self._head_size)

        _q = apply_rotary_emb(_q, freq_cis[start_pos:start_pos + _seq])
        _k = apply_rotary_emb(_k, freq_cis[start_pos:start_pos + _seq])

        # 处理 past_key_value
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k is not None and past_v is not None:  # 仅当两者都是有效张量时才拼接
                _k = torch.cat([past_k, _k], dim=1)
                _v = torch.cat([past_v, _v], dim=1)

        # 如果使用缓存，则准备返回的 present_key_value
        present_key_value = None
        if use_cache:
            present_key_value = (_k, _v)

        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)

        #此时还得用老的代码，训练的模型形状不对
        _k = _k[:, None].repeat(1, self._group, 1, 1, 1).reshape(
            _bn, -1, start_pos + _seq, self._head_size)
        _v = _v[:, None].repeat(1, self._group, 1, 1, 1).reshape(
            _bn, -1, start_pos + _seq, self._head_size)
        # 处理 GQA: 将 kv heads 复制到 q heads
        # _k = _k[:, :, None].repeat(1, 1, self._group, 1, 1).reshape(
        #     _bn, self._n_q_heads, -1, self._head_size)
        # _v = _v[:, :, None].repeat(1, 1, self._group, 1, 1).reshape(
        #     _bn, self._n_q_heads, -1, self._head_size)

        # _causul = torch.ones(_seq, _seq)
        # _causul = torch.triu(_causul, diagonal=1)
        # _causul[_causul == 1] = -torch.inf
        # _causul = _causul.to(x.device)

        # _score = _q@_k.permute(0, 1, 3, 2)/_dk
        # print(_score.shape)
        # _score = torch.softmax(_score+_causul, dim=-1)

        # _o = _score@_v
        # print(_q.shape, _k.shape, _v.shape)

        if start_pos == 0:
            _o = F.scaled_dot_product_attention(
                _q, _k, _v, attn_mask=None, is_causal=True)
        else:
            _o = F.scaled_dot_product_attention(
                _q, _k, _v, attn_mask=None, is_causal=False)

        _o = _o.permute(0, 2, 1, 3)
        _o = _o.reshape(_bn, _seq, -1)

        output = self._ow(_o)
        return output, present_key_value

class FFN(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self._w0=nn.Linear(input_dim,hidden_dim)
        self._w1=nn.Linear(input_dim,hidden_dim)
        self._w2=nn.Linear(hidden_dim,input_dim)
        self._gate=nn.SiLU()

    def forward(self,input):
        return self._w2(self._w0(input) * self._gate(self._w1(input)))

class RMSNorm(nn.Module):
    def __init__(self,input_dim,eps=1e-6):
        super().__init__()
        self._w=nn.Parameter(torch.randn(input_dim))
        self.eps=eps

    def forward(self,input):
        # return self._w * input/(torch.norm(input, p=2)+self.eps)#输入为0向量时,L2范数为0,在显卡上触发除0bug
        return self._w * input * torch.rsqrt(input.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class StorierConfig(PretrainedConfig):
    model_type = "storier"  # 自定义模型类型

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.num_hidden_layers = self.num_layers = kwargs.get("num_layers")  # 会将两个值都写入config.json影响阅读和配置
        # self.hidden_size = self.input_dim = kwargs.get("input_dim")
        # self.intermediate_size = self.hide_dim = kwargs.get("hide_dim")
        # self.num_attention_heads = self.n_q_heads = kwargs.get("n_q_heads")
        # self.num_key_value_heads = self.n_kv_heads = kwargs.get("n_kv_heads")
        # self.max_position_embeddings = self.max_pos_len = kwargs.get("max_pos_len")
        self.num_layers = kwargs.get("num_layers")
        self.input_dim = kwargs.get("input_dim")
        self.hide_dim = kwargs.get("hide_dim")
        self.n_q_heads = kwargs.get("n_q_heads")
        self.n_kv_heads = kwargs.get("n_kv_heads")
        self.max_pos_len = kwargs.get("max_pos_len")
        self.vocab_size = kwargs.get("vocab_size")
        # 可选：是否默认使用缓存
        self.use_cache = kwargs.get("use_cache", True)

        self.pad_token_id = 0
        self.bos_token_id = 2
        self.eos_token_id = 3

        self.auto_map = {
            "AutoModelForCausalLM": "model.StorierModel",
            "AutoConfig": "model.StorierConfig"
        }

    # 添加以下 property，将自定义字段映射为标准名称
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


class StorierModel(PreTrainedModel,GenerationMixin):
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
        # 确定 use_cache 默认值
        if use_cache is None:
            use_cache = self.config.use_cache if hasattr(self.config, 'use_cache') else True

        # 1. 转换 past_key_values 为兼容格式（列表 of (k, v)）
        if past_key_values is not None:
            if hasattr(past_key_values, 'key_cache'):  # 输入是 DynamicCache
                key_cache, value_cache = past_key_values.to_legacy_cache()
                # 处理可能存在的 None 缓存（例如首次生成时缓存为空）
                past_kv_list = []
                for i in range(len(key_cache)):
                    if key_cache[i] is not None and value_cache[i] is not None:
                        past_kv_list.append((key_cache[i], value_cache[i]))
                    else:
                        past_kv_list.append(None)  # 保持 None 占位
            else:
                # 输入已经是列表格式
                past_kv_list = past_key_values
        else:
            past_kv_list = None

        # 计算 start_pos
        start_pos = 0
        if past_key_values is not None:
            if hasattr(past_key_values, 'get_seq_length'):
                start_pos = past_key_values.get_seq_length()
            elif isinstance(past_key_values, list) and past_key_values[0] is not None:
                start_pos = past_key_values[0][0].shape[1]

        tokens = self.emb(input_ids)
        features, present_key_values = self.tf_layer(
            tokens, start_pos, past_kv_list, use_cache)
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
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=None,
            attentions=None,
        )

    def _set_gradient_checkpointing(self, module, value=False):
        """实现梯度检查点的启用/禁用"""
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, value):
        self.emb = value

    @staticmethod
    def _tsoftmax(xs, temp):
        _o = xs - xs.mean()
        return torch.exp(_o / temp) / (torch.exp(_o / temp).sum(-1) + 1e-5)


AutoConfig.register("storier", StorierConfig)
modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["storier"] = "StorierModel"

if __name__ == '__main__':
    config = StorierConfig(num_layers=48,
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
    model.load_state_dict(torch.load(
        f"mp_rank_00_model_states.pt", weights_only=False)["module"])

    # 保存模型
    save_directory = "./cache/storier"
    model.save_pretrained(save_directory)

    shutil.copy2("model.py",save_directory)

    print("Model and configuration saved successfully.")

    config_dict = config.to_dict()
    config_dict["auto_map"] = {
        "AutoModelForCausalLM":"model.StorierModel",
        "AutoConfig":"model.StorierConfig",
    }

    config_file = os.path.join(save_directory, "config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
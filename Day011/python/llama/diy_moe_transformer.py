import torch
from torch import nn
from Day011.python.transformer.diy_rope import *

class TransformerDecoder(nn.Module):
    """
        解码器
    """

    def __init__(self,
                 num_layers,  # 解码器的层数
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 num_experts,
                 top_k,
                 max_len
                 ):
        super().__init__()

        self._layers = nn.ModuleList(
            [TransformerLayer(input_dim,
                              hide_dim,
                              n_q_heads,
                              n_kv_heads,
                              num_experts,
                              top_k) for _ in range(num_layers)]
        )

        _freq_cis = precompute_freqs_cis(input_dim // n_q_heads, max_len)

        self.register_buffer("freq_cis", _freq_cis, persistent=False)

    def forward(self, x):
        _x = x
        for _layer in self._layers:
            _x = _layer(_x, self.freq_cis)
        return _x


# class Expert(nn.Module):

#     def __init__(self,
#                  num_experts,
#                  top_k,
#                  input_dim,
#                  hide_dim):
#         super().__init__()

#         self._top_k = top_k
#         self._experts = nn.ModuleList(
#             [FFN(input_dim, hide_dim) for _ in range(num_experts)])
#         self._gate = nn.Linear(input_dim, num_experts)

#     def forward(self, x):
#         _bn, _seq, _vec = x.shape

#         _gate_out = self._gate(x)

#         _top_values, _top_indices = torch.topk(_gate_out, self._top_k, dim=-1)

#         _output = torch.zeros(_bn, _seq, self._top_k, _vec).to(x.device)

#         for _i in range(_bn):
#             for _j in range(_seq):
#                 for _k in range(self._top_k):
#                     _expert = self._experts[_top_indices[_i, _j, _k]]
#                     _output[_i, _j, _k] = _expert(x[_i, _j])

#         # return _output.mean(dim=2)

#         w = torch.softmax(_top_values, dim=-1)[:, :, :, None]
#         return (w*_output).sum(dim=2)

class Expert(nn.Module):

    def __init__(self, num_experts, top_k, input_dim, hide_dim):
        super().__init__()

        self.top_k = top_k

        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [FFN(input_dim, hide_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, inputs):
        _inputs = inputs.reshape(-1, inputs.shape[-1])
        gate_logits = self.gate(_inputs)

        weights, selected_experts = torch.topk(gate_logits, self.top_k)
        weights = torch.softmax(weights, dim=-1)
        results = torch.zeros_like(_inputs)

        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            _v = weights[batch_idx, nth_expert, None] * expert(_inputs[batch_idx])
            results[batch_idx] += _v

        results = results.reshape(*inputs.shape)
        return results


class TransformerLayer(nn.Module):
    """
    单层的Transformer结构
    """

    def __init__(self,
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 num_experts,
                 top_k):
        super().__init__()

        self._att_norm = RMSNorm(input_dim)
        self._att_layer = MultiHeadAttention(input_dim, n_q_heads, n_kv_heads)
        self._ffn_norm = RMSNorm(input_dim)
        self._ffn_layer = Expert(num_experts, top_k, input_dim, hide_dim)

    def forward(self, x, freq_cis):
        _x = x
        _x = self._att_norm(_x)
        _x = self._att_layer(_x, freq_cis)

        _x = x + _x

        _y = _x
        _y = self._ffn_norm(_y)
        _y = self._ffn_layer(_y)

        _y = _y + _x

        return _y


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 input_dim,
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
        self._ow = nn.Linear(input_dim, input_dim)

    def forward(self, x, freq_cis):
        _bn, _seq, _ = x.shape
        _dk = self._head_size ** 0.5

        _q, _k, _v = self._qw(x), self._kw(x), self._vw(x)

        _q = _q.reshape(_bn, _seq, self._n_q_heads, self._head_size)
        _k = _k.reshape(_bn, _seq, self._n_kv_heads, self._head_size)
        _v = _v.reshape(_bn, _seq, self._n_kv_heads, self._head_size)

        _q = apply_rotary_emb(_q, freq_cis[:_seq])
        _k = apply_rotary_emb(_k, freq_cis[:_seq])

        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)

        _causul = torch.ones(_seq, _seq)
        _causul = torch.triu(_causul, diagonal=1)
        _causul[_causul == 1] = -torch.inf
        _causul = _causul.to(x.device)

        _k = _k[:, None].repeat(1, self._group, 1, 1, 1).reshape(
            _bn, -1, _seq, self._head_size)
        _v = _v[:, None].repeat(1, self._group, 1, 1, 1).reshape(
            _bn, -1, _seq, self._head_size)

        _score = _q @ _k.permute(0, 1, 3, 2) / _dk
        _score = torch.softmax(_score + _causul, dim=-1)

        _o = _score @ _v

        _o = _o.permute(0, 2, 1, 3)
        _o = _o.reshape(_bn, _seq, -1)

        return self._ow(_o)

class GQAAttention(nn.Module):#GQA版本，
    def __init__(self,input_dim,q_heads,kv_heads,_s,max_batch_size,max_seq_len):#_s和input的s长度一样
        super().__init__()
        assert q_heads % kv_heads == 0
        self._q_heads = q_heads
        self._kv_heads = kv_heads

        self._head_size = input_dim // self.q_heads#GQA
        self._group = q_heads // kv_heads

        self._qw = nn.Linear(input_dim,self._head_size*self._q_heads)
        self._kw = nn.Linear(input_dim,self._head_size*self._kv_heads)
        self._vw = nn.Linear(input_dim,self._head_size*self._kv_heads)
        self._ow = nn.Linear(input_dim,input_dim)

        # _causul = torch.ones(_s, _s)
        # _causul = torch.triu(_causul, diagonal=1)
        # _causul[_causul == 1] = -torch.inf
        # self.register_buffer("causul",_causul,persistent=False)#不同于net.cuda(),单独传到显存中(不作为需要反向求导修改的参数),persistent不保存进模型

        self.register_buffer(
            "cache_k",
            torch.zeros(max_batch_size, max_seq_len, kv_heads, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(max_batch_size, max_seq_len, kv_heads, self.head_dim),
            persistent=False,
        )

    def forward(self,input,freqs_cis,start_pos):
        _n,_s,_v=input.shape
        # _h=_v//self._heads
        _dk=self._head_size ** 0.5

        _q, _k, _v= self._qw(input),self._kw(input),self._vw(input)

        _q, _k, _v=_q.reshape(_n,_s,self._q_heads,self._head_size),_k.reshape(_n,_s,self._kv_heads,self._head_size),_v.reshape(_n,_s,self._kv_heads,self._head_size)

        _q, _k, _v=_q.permute(0,2,1,3),_k.permute(0,2,1,3),_v.permute(0,2,1,3)

        _q = apply_rotary_emb(_q,freqs_cis[:_s])#将4096*2截断
        _k = apply_rotary_emb(_k,freqs_cis[:_s])#将4096*2截断

        self.cache_k[:_n, start_pos: start_pos + _s] = _k
        self.cache_v[:_n, start_pos: start_pos + _s] = _v

        _k = self.cache_k[:_n, : start_pos + _s]
        _v = self.cache_v[:_n, : start_pos + _s]

        _causul = torch.ones(_s, _s)
        _causul = torch.triu(_causul, diagonal=1)
        _causul[_causul == 1] = -torch.inf
        _causul=_causul.to(input.device)

        _k = _k[:,None].repeat(1,self._group,1,1,1).reshape(_n,-1,_s,self._head_size)
        _v = _v[:,None].repeat(1,self._group,1,1,1).reshape(_n,-1,_s,self._head_size)

        _score = torch.softmax(_q@_k.transpose(-2,-1)/_dk+_causul,dim=-1)#NHSS
        # _score = torch.softmax(_q@_k.transpose(-2,-1)/_dk+self.causul,dim=-1)#NHSS

        _o = (_score @ _v).permute(0,2,1,3)

        _o = _o.reshape(_n,_s, -1)
        return self._ow(_o)

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
        return self._w * input/(torch.norm(input, p=2)+self.eps)

if __name__ == '__main__':
    bn = 3
    seq = 7
    vec = 128
    n_q_heads = 4
    n_kv_heads = 2
    n_layers = 2
    max_len = seq * 2
    num_experts = 4
    top_k = 2

    freq_cis = precompute_freqs_cis(vec // n_q_heads, max_len)

    x = torch.randn(bn, seq, vec)
    decoder = TransformerDecoder(
        num_layers=n_layers,
        input_dim=vec,
        hide_dim=vec // 2,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        num_experts=num_experts,
        top_k=2,
        max_len=max_len
    )

    y = decoder(x)
    print(y.shape)
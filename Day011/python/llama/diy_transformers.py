import torch
from torch import nn
from Day011.python.transformer.diy_rope import *

class TransformerEncoder(nn.Module):
    def __init__(self,layer_nums):
        pass
    def forward(self,input):
        pass

class TransformerDecoder(nn.Module):
    def __init__(self,layer_nums,input_dim,hidden_dim,heads,max_len):
        super().__init__()
        self._layers = nn.ModuleList([
            TransformerLayer(input_dim,hidden_dim,heads) for _ in range(layer_nums)
        ])

        _freqs_cis = precompute_freqs_cis(input_dim//heads,4096*2)
        self.register_buffer("freqs_cis",_freqs_cis,persistent=False)

    def forward(self,x):
        _x=x
        for layer in self._layers:
            _x = layer(_x,self.freqs_cis)
        return _x


class TransformerLayer(nn.Module):
    def __init__(self,input_dim,hidden_dim,heads):
        super().__init__()
        self.attention_layer = MultiHeadAttention(input_dim,heads)
        self.ffn = FFN(input_dim,hidden_dim)
        self.attention_norm = RMSNorm(input_dim)
        self.ffn_norm = RMSNorm(input_dim)

    def forward(self,input,freqs_cis):
        _x=self.attention_norm(input)
        _x=self.attention_layer(_x,freqs_cis)
        _x=_x+input

        intermediate_output = _x

        _x=self.ffn_norm(_x)
        _x=self.ffn(_x)

        output=_x+intermediate_output
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self,input_dim,q_heads,kv_heads,_s):#_s和input的s长度一样
        super().__init__()
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

    def forward(self,input,freqs_cis):
        _n,_s,_v=input.shape
        # _h=_v//self._heads
        _dk=self._head_size ** 0.5

        _q, _k, _v= self._qw(input),self._kw(input),self._vw(input)

        _q, _k, _v=_q.reshape(_n,_s,self._q_heads,self._head_size),_k.reshape(_n,_s,self._kv_heads,self._head_size),_v.reshape(_n,_s,self._kv_heads,self._head_size)

        _q, _k, _v=_q.permute(0,2,1,3),_k.permute(0,2,1,3),_v.permute(0,2,1,3)

        _q = apply_rotary_emb(_q,freqs_cis[:_s])#将4096*2截断
        _k = apply_rotary_emb(_k,freqs_cis[:_s])#将4096*2截断

        _causul = torch.ones(_s, _s)
        _causul = torch.triu(_causul, diagonal=1)
        _causul[_causul == 1] = -torch.inf
        _causul=_causul.to(input.device)

        _k = _k[:,None].repeat(1,self._group,1,1,1).reshape(_n,_s,self._head_size)
        _v = _v[:,None].repeat(1,self._group,1,1,1).reshape(_n,_s,self._head_size)

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

import torch
from torch import nn
from Day011.python.transformer.diy_rope import *

class TransformerEncoder(nn.Module):
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

class Expert(nn.Module):
    def __init__(self,num_experts,top_k,input_dim,hidden_dim):
        super().__init__()
        self._experts = nn.ModuleList([FFN(input_dim,hidden_dim) for _ in range(num_experts)])
        self._gate = nn.Linear(input_dim,num_experts)
        self._top_k = top_k

    def forward(self,x):#NSV
        _n,_s,_v=x.shape
        _gate_out = self._gate(x)#NSnum
        _top_values,_top_indices = torch.topk(_gate_out,self.top_k,dim=-1)#NSk,讲的不好

        _out=torch.zeros(_n,_s,self._top_k,_v)

        for i in range(_n):
            for j in range(_s):
                for k in range(self.top_k):
                    _out[i,j,k] = self._experts[_top_indices[i,j,k]](x[i,j])
        #_out,NS*topk*seld.hidden_dim
        w = torch.softmax(_top_values,dim=-1)[:,:,:,None]#NSK1

        return (w*_out).sum(dim=2)#对应位置相乘相加,返回的依然是hidden_dim，而非input_dim

        # return _out.mean(dim=2)


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
    def __init__(self,input_dim,heads):
        super().__init__()
        self._heads = heads
        self._qw = nn.Linear(input_dim,input_dim)
        self._kw = nn.Linear(input_dim,input_dim)
        self._vw = nn.Linear(input_dim,input_dim)
        self._ow = nn.Linear(input_dim,input_dim)

    def forward(self, input,freqs_cis,start_pos):
        _n, _s, _v = input.shape
        _h = _v // self._heads
        _dk = _h ** 0.5

        _q,_k,_v = self._qw(input),self._kw(input),self._vw(input)

        _q = _q.reshape(_n,_s,self._heads,_h).permute(0,2,1,3)
        _k = _k.reshape(_n,_s,self._heads,_h).permute(0,2,1,3)
        _v = _v.reshape(_n,_s,self._heads,_h).permute(0,2,1,3)

        _q = apply_rotary_emb(_q, freqs_cis[:_s])  # 将4096*2截断
        _k = apply_rotary_emb(_k, freqs_cis[:_s])  # 将4096*2截断

        _score = torch.softmax(_q @ _k.transpose(-2, -1) / _dk , dim=-1)  # NHSS
        _o = (_score @ _v).permute(0, 2, 1, 3)
        _o = _o.reshape(_n, _s, -1)

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

        self.cache_k = torch.zeros(
            (
                max_batch_size,
                max_seq_len,
                kv_heads,
                self._head_size,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                max_batch_size,
                max_seq_len,
                kv_heads,
                self._head_size,
            )
        ).cuda()

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

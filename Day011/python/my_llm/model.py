import torch
from torch import nn
from torch.nn import init
from diy_transformers import TransformerDecoder

class Storier(nn.Module):
    def __init__(self,num_layers=48,input_dim=768,hide_dim=3072,n_q_heads=8,n_kv_heads=4,max_len=16384,num_vocs=50000,cache_max_batch_size=None,cache_max_seq_len=None):
        super().__init__()
        self.emb=nn.Embedding(num_vocs,input_dim)
        self.cache_max_batch_size = cache_max_batch_size
        self.tf_layer=TransformerDecoder(num_layers=num_layers,input_dim=input_dim,hide_dim=hide_dim,n_q_heads=n_q_heads,n_kv_heads=n_kv_heads,max_len=max_len,cache_max_batch_size=cache_max_batch_size,cache_max_seq_len=cache_max_seq_len)

        # self.output_layer=nn.Linear(input_dim,num_vocs,bias=False)
        # self.output_layer.weight=self.emb.weight
        self.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
            # m.reset_parameters()
        elif isinstance(m, nn.Embedding):
            init.xavier_uniform_(m.weight)
            # m.reset_parameters()

    def _forward(self,ids,start_pos):
        assert ids is not None and len(ids) > 0

        token=self.emb(ids)
        feature=self.tf_layer(token,start_pos)
        return feature @ self.emb.weight.T#不加softmax,与自身embedding做损失,不是求最大的索引项
        # return self.output_layer(feature)

    def forward(self, ids, start_pos):
        if self.cache_max_batch_size is None:
            return self._forward(ids, 0)
        else:
            with torch.no_grad():
                return self._forward(ids, start_pos)


if __name__ == '__main__':
    storier=Storier(num_layers=2,input_dim=128,hide_dim=96,n_q_heads=2,n_kv_heads=1,max_len=1024,num_vocs=50000,cache_max_batch_size=None,cache_max_seq_len=None)
    x = torch.randint(0, 100, size=(3, 8))
    print(x)
    print(storier(x,0).shape)

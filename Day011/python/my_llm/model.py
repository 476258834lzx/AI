import torch
from torch import nn
from diy_transformers import TransformerDecoder

class Storier(nn.Module):
    def __init__(self,num_layers=48,input_dim=768,hide_dim=3072,n_q_heads=8,n_kv_heads=4,max_len=16384,num_vocs=50000,cache_max_batch_size=None,cache_max_seq_len=None):
        super().__init__()
        self.emb=nn.Embedding(num_vocs,input_dim)
        self.cache_max_batch_size = cache_max_batch_size
        self.tf_layer=TransformerDecoder(num_layers=num_layers,input_dim=input_dim,hide_dim=hide_dim,n_q_heads=n_q_heads,n_kv_heads=n_kv_heads,max_len=max_len,cache_max_batch_size=cache_max_batch_size,cache_max_seq_len=cache_max_seq_len)

        # self.output_layer=nn.Linear(input_dim,num_vocs,bias=False)
        # self.output_layer.weight=self.emb.weight

    def _forward(self,ids,start_pos):
        assert ids is not None and len(ids) > 0

        token=self.emb(ids)
        feature=self.tf_layer(token,start_pos)
        return feature @ self.emb.weight.T
        # return self.output_layer(feature)

    def forward(self, ids, start_pos):
        if self.cache_max_batch_size is None:
            return self._forward(ids, 0)
        else:
            with torch.no_grad():
                return self._forward(ids, start_pos)


if __name__ == '__main__':
    storier=Storier()
    x = torch.randint(0, 100, size=(3, 8))
    print(x)
    print(storier(x,0).shape)

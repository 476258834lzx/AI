import torch
from torch import nn
from diy_transformers import TransformerDecoder

class Storier(nn.Module):
    def __init__(self,num_layers=48,input_dim=1024,hide_dim=683,n_q_heads=8,n_kv_heads=4,max_len=1024,num_vocs=50000):
        super().__init__()
        self.emb=nn.Embedding(num_vocs,input_dim)
        self.tf_layer=TransformerDecoder(num_layers=num_layers,input_dim=input_dim,hide_dim=hide_dim,n_q_heads=n_q_heads,n_kv_heads=n_kv_heads,max_len=max_len)
        self.output_layer=nn.Linear(input_dim,num_vocs,bias=False)
        self.output_layer.weight=self.emb.weight

    def forward(self,ids):
        token=self.emb(ids)
        feature=self.tf_layer(token)
        return self.output_layer(feature)

if __name__ == '__main__':
    storier=Storier()
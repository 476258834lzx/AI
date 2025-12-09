import torch
from torch import nn
from diy_transformerencoder import TransformerEncoder

class VIT(nn.Module):
    def __init__(self):
        super().__init__()
        # self._emb_layer=nn.Conv2d(1,64,14,14,bias=False)
        self._emb_layer=nn.Linear(14*14,64,bias=False)

        self._tf_layer=TransformerEncoder(num_layers=2,input_dim=64,hide_dim=48,n_q_heads=2,n_kv_heads=1,max_len=4)
        self._out_layer=nn.Linear(64,10,bias=False)

        _cls_token=torch.randn(64)
        self.register_buffer('_cls_token',_cls_token)

    def forward(self,input):
        n,c,h,w=input.shape
        # _x = self._emb_layer(input).reshape(n,c,-1).permute(0,2,1)
        _x=input.reshape(n,c,2,h//2,2,w//2).permute(0,2,4,1,3,5).reshape(n,4,-1)

        token=self._emb_layer(_x)
        _cls_token=self._cls_token[None,None].repeat(n,1,1)
        token=torch.concat([_cls_token,token],dim=1)

        features=self.tf_layer(token)
        feature=features[:,0]
        return self._out_layer(feature)

if __name__ == '__main__':
    vit=VIT()
import torch
from torch import nn

if __name__ == '__main__':
    x=torch.randn(3,10,4)
                                                                 #批次为第一个维度(NSV),False时为SNV
    encoder_layer=nn.TransformerEncoderLayer(d_model=4,nhead=2,batch_first=True)
    transformer_encoder=nn.TransformerEncoder(encoder_layer,num_layers=2)
    y=encoder_layer(x)
    print(x.shape)


    decoder_layer=nn.TransformerDecoderLayer(d_model=4,nhead=2)
    tgt=torch.randn(3,10,4)#反馈单元
    z=decoder_layer(tgt,x)
    print(z.shape)
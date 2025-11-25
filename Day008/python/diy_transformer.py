import torch
from torch import nn
from attention import Attention

class CBM(nn.Module):
    def __init__(self,input_channel):
        super(CBM, self).__init__()
        self._cbm=nn.Sequential(
            nn.Conv1d(input_channel, input_channel, (3,), (1,),(1,)),
            nn.BatchNorm1d(input_channel),
            nn.Mish(),
            # nn.Dropout1d()
        )

    def forward(self,x):
        return self._cbm(x)

class Transformer(nn.Module):
    def __init__(self,head,input_channel,input_dim):
        super(Transformer, self).__init__()
        self.input_dim=input_dim
        self._att=Attention(head,input_dim,input_dim)
        self.fc=CBM(input_channel)

    def forward(self,x):
        if self.input_dim==1:
            y=self._att.single(x,x,x)
        else:
            y=self._att(x,x,x)

        y=x+y
        z=self.fc(y)
        z=y+z
        return z

if __name__ == '__main__':
    x=torch.randn(3,10,4)
    transformer=Transformer(2,10,4)
    y=transformer(x)
    print(y.shape)
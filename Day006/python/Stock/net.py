import torch
from torch import nn

class Net(nn.Module):
    def __init__(self,input_dim):
        super(Net, self).__init__()
        self._map_layer=nn.Sequential(
            nn.Conv1d(input_dim,16,(3,),(1,))
        )
        self._encoder_layer=nn.TransformerEncoderLayer(d_model=16,nhead=2,batch_first=True)
        self._transformer_encoder=nn.TransformerEncoder(self._encoder_layer,num_layers=6)

        self._output_layer=nn.Sequential(
            nn.Linear(16,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        _x=x.permute(0,2,1)
        _y=self._map_layer(_x)
        _y=_y.permute(0,2,1)
        _y=self._transformer_encoder(_y)
        _y=_y[:,-1]
        _y=self._output_layer(_y)

        return _y

if __name__ == '__main__':
    x=torch.randn(2,5,7)
    net=Net(7)
    y=net(x)
    print(y.shape)
from torch import nn
import torch
from torch.utils.data import DataLoader

class NetV1(nn.Module):
    def __init__(self):
        super(NetV1, self).__init__()
        self.layer =nn.Sequential(
            nn.Conv2d(1,2,(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 4, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(4, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3))
        )
        self.out_layer=nn.Sequential(
            nn.Linear(64*3*3,10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        out=self.layer(x)
        #NCHW->NV
        out=out.reshape(-1,64*3*3)
        out=self.out_layer(out)
        return out


if __name__ == '__main__':
    net=NetV1()
    x = torch.randn(1,1,28,28)
    print(net(x).shape)
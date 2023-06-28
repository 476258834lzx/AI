import torch
from torch import nn

class NetV1(nn.Module):
    def __init__(self):
        super(NetV1, self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        out=self.layer(x)
        return out
if __name__ == '__main__':
    net=NetV1()
    x=torch.randn(1,784)
    y=net(x)
    print(y)
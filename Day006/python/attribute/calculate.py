import thop,torch

from torch import nn

# conv=nn.Conv2d(3,16,3,1)
# x=torch.randn(1,3,16,16)
# flops,params=thop.profile(conv,(x,))
# print(flops,params)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(3, 16, 3, 1)
        )
    def forward(self,x):
        return self.layers(x)

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(3*16*16,16*14*14)
        )
    def forward(self,x):
        return self.layers(x)

if __name__ == '__main__':
    net=Net()
    x = torch.randn(1, 3, 16, 16)
    flops,params=thop.profile(net,(x,))
    print(flops,params)
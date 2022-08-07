import torch
from torch import nn

class NetV1(nn.Module):
    def __init__(self):
        super(NetV1, self).__init__()
        self.w=nn.Parameter(torch.randn(784,10))
    def forward(self,x):
    #(N*10)=(N*784)@(784*10)
        h=x@self.w
        #softmax(e的xi次方除以e的j次方求和)
    #   (N*10)=(N*10)/(N*1)广播机制N*1广播到N*10，标量也可以广播到N*10
        print(torch.exp(h).shape)
        print(torch.sum(h,dim=1,keepdim=True).shape)
        h=torch.exp(h)/torch.sum(h,dim=1,keepdim=True)
        return h


class NetV2(nn.Module):
    def __init__(self):
        super(NetV2, self).__init__()
        self.fc1 = nn.Linear(784,100)
        self.ac1=nn.PReLU()
        self.fc2=nn.Linear(100,10)
        self.ot1=nn.Softmax(dim=1)
    def forward(self, x):
        h=self.fc1(x)
        h=self.ac1(h)
        h=self.fc2(h)
        out=self.ot1(h)
        return out

class NetV3(nn.Module):
    def __init__(self):
        super(NetV3, self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(784,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        out=self.layer(x)
        return out

if __name__ == '__main__':
    net=NetV3()
    x=torch.randn(3,784)
    print(net(x))
from torch import nn
from torch import optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Liner_layers=nn.Sequential(
            nn.Linear(784,512),
            nn.BatchNorm1d(512),#批次
            nn.ReLU(),
            nn.Dropout(0.2),#随机当前层20%神经元失活
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,10)
        )
        self.conv2d_layers=nn.Sequential(
            nn.Conv2d(3,32,(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(32,64,(3,3)),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(64,128,(3,3))
        )
        self.res_block=nn.Sequential(
            nn.Conv2d(16,16,(3,3),(3,3),(1,1),bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (3, 3), (1, 1))
        )
    def forward(self,x):
        return self.res_block(x)+x

net=Net()
opt=optim.Adam(net.parameters(),weight_decay=0.2)

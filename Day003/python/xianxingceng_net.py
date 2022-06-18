import torch
import os
from torch import nn
import matplotlib.pyplot as plt
from torch import optim
os.environ['KMP_DUPLICATE_LIB_OK']='True'
xs=torch.unsqueeze(torch.arange(0.01,1,0.01),dim=1)
ys=3*xs+4


class line(torch.nn.Module):
    def __init__(self):
        super(line, self).__init__()
        self.layer1=nn.Linear(1,20)
        self.layer2=nn.Linear(20,64)
        self.layer3=nn.Linear(64,128)
        self.layer4=nn.Linear(128,64)
        self.layer5=nn.Linear(64,1)

    def forward(self,x):
        fc1=self.layer1(x)
        fc2=self.layer2(fc1)
        fc3=self.layer3(fc2)
        fc4=self.layer4(fc3)
        fc5=self.layer5(fc4)
        return fc5

if __name__ == '__main__':
    net=line()
    opt=optim.Adam(net.parameters())
    loss_func=nn.MSELoss()

    plt.ion()
    for epoch in range(1000):
        out=net(xs)
        loss=loss_func(out,ys)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch%5==0:
            plt.cla()
            plt.plot(xs, ys, '.')
            plt.plot(xs, out.detach())
            plt.title("loss%.4f"%loss.item())
            plt.pause(0.001)
    plt.ioff()
    plt.show()
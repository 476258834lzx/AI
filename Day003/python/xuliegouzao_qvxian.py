import random

import torch
import os
from torch import nn
import matplotlib.pyplot as plt
from torch import optim
os.environ['KMP_DUPLICATE_LIB_OK']='True'
xs=torch.unsqueeze(torch.arange(0.01,1,0.01),dim=1)###
ys=[e.pow(3)*random.randint(1,6) for e in xs]
ys=torch.stack(ys)###

class line(torch.nn.Module):
    def __init__(self):
        super(line, self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(1,20),
            nn.Sigmoid(),
            nn.Linear(20,64),
            nn.Sigmoid(),
            nn.Linear(64,128),
            nn.Sigmoid(),
            nn.Linear(128,64),
            nn.Sigmoid(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        out=self.layer(x)
        return out

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
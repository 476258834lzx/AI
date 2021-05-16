import torch
from torch import optim
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
x=torch.arange(0,1,0.01)
y=3*x+4+torch.rand(100)

class Line(torch.nn.Module):
    def __init__(self):
        super(Line, self).__init__()
        self.w=torch.nn.Parameter(torch.rand(1))
        self.b=torch.nn.Parameter(torch.rand(1))

    def forward(self,x):
        z=self.w*x+self.b
        return z
if __name__ == '__main__':
    line=Line()
    opt=optim.SGD(line.parameters(),lr=0.1)
    plt.ion()
    loss_fn=torch.nn.MSELoss()
    for epoch in range(30):
        for _x,_y in zip(x,y):
            z=line(_x)
            # loss=(z-_y)**2
            loss=loss_fn(z,_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(line.w,line.b,loss)
        plt.cla()
        plt.plot(x,y,".")
        v=[line.w*e+line.b for e in x]
        plt.plot(x,v)
        plt.pause(0.01)
    plt.ioff()
    plt.show()
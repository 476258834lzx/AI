import torch
from torch import nn

config=[
    [-1,32,1,2],
    [1,16,1,1],
    [6,24,2,2],
    [6,32,3,2],
    [6,64,4,2],
    [6,96,3,1],
    [6,160,3,2],
    [6,320,1,1]
]

class bottleneck_block(nn.Module):
    def __init__(self,c_in,i,t,c,n,s):
        super(bottleneck_block, self).__init__()
        #重复最后一次才调用步长为2的下采样
        self.n=n
        self.i=i

        _s = s if self.i==self.n-1 else 1#n为重复次数，循环索引从0开始，最后一次步长为2,通道数变为c_out
        _c=c if self.i==n-1 else c_in

        _c_in=c_in*t#升通道倍数

        self.layer=nn.Sequential(
            nn.Conv2d(c_in,_c_in,(1,1),(1,1),bias=False),
            nn.BatchNorm2d(_c_in),
            nn.ReLU6(),
            nn.Conv2d(_c_in,_c_in,(3,3),(_s,_s),padding=1,groups=_c_in,bias=False),
            nn.BatchNorm2d(_c_in),
            nn.ReLU6(),
            nn.Conv2d(_c_in,_c,(1,1),(1,1),bias=False),
            nn.BatchNorm2d(_c)
        )

    def forward(self,x):
        if self.i==self.n-1:
            return self.layer(x)
        else:
            return self.layer(x)+x

class Mobilenetv2(nn.Module):
    def __init__(self,config):
        super(Mobilenetv2, self).__init__()

        self.blocks=[]
        c_in=config[0][1]
        for t,c,n,s in config[1:]:
            for i in range(n):
                self.blocks.append(bottleneck_block(c_in,i,t,c,n,s))
            c_in=c

        self.layer=nn.Sequential(
            nn.Conv2d(3,32,(3,3),(2,2),(1,1),bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            *self.blocks,
            nn.Conv2d(320,1280,(1,1),(1,1),bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            nn.AvgPool2d((7,7),1),
            nn.Conv2d(1280,10,(1,1),(1,1),bias=False)
        )
    def forward(self,x):
        return self.layer(x)

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    net=Mobilenetv2(config)
    print(net(x).shape)
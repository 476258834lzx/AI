import torch
from torch.nn import functional as F
from torch import nn

class CNNLayer(nn.Module):
    def __init__(self,c_in,c_out):
        super(CNNLayer, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(c_in,c_out,(3,3),(1,1),(1,1),padding_mode="reflect",bias=False),
            nn.BatchNorm2d(c_out),
            nn.Dropout2d(0.3),#分割数据标签难以获取，数据量过小易过拟合
            nn.LeakyReLU(),
            nn.Conv2d(c_out, c_out//2, (3, 3), (1, 1), (1, 1), padding_mode="reflect", bias=False),
            nn.BatchNorm2d(c_out//2),
            nn.Dropout2d(0.4),
            nn.LeakyReLU(),
            nn.Conv2d(c_out//2, c_out, (3, 3), (1, 1), (1, 1), padding_mode="reflect", bias=False),
            nn.BatchNorm2d(c_out),
            nn.Dropout2d(0.4),
            nn.LeakyReLU()
        )

    def forward(self,x):
        return self.layer(x)

class Downsampling(nn.Module):
    def __init__(self,c):
        super(Downsampling, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(c,c,(3,3),(2,2),(1,1),padding_mode="reflect",bias=False),
            nn.LeakyReLU()
        )

    def forward(self,x):
        return self.layer(x)

class Upsampling(nn.Module):
    def __init__(self,c):
        super(Upsampling, self).__init__()
        self.layer=nn.Conv2d(c,c//2,(1,1),(1,1),bias=False)

    def forward(self,x,r):
        up=F.interpolate(x,scale_factor=2,mode="nearest")
        out=self.layer(up)
        return torch.cat((out,r),1)

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.C1=CNNLayer(3,64)
        self.D1=Downsampling(64)
        self.C2 = CNNLayer(64, 128)
        self.D2 = Downsampling(128)
        self.C3 = CNNLayer(128, 256)
        self.D3 = Downsampling(256)
        self.C4 = CNNLayer(256, 512)
        self.D4 = Downsampling(512)
        self.C5 = CNNLayer(512, 1024)
        self.U1 = Upsampling(1024)
        self.C6 = CNNLayer(1024, 512)
        self.U2 = Upsampling(512)
        self.C7 = CNNLayer(512, 256)
        self.U3 = Upsampling(256)
        self.C8 = CNNLayer(256, 128)
        self.U4 = Upsampling(128)
        self.C9 = CNNLayer(128, 64)
        self.pre=nn.Conv2d(64,3,(3,3),(1,1),(1,1))
        # self.Th=nn.Sigmoid()

    def forward(self,x):
        R1=self.C1(x)
        R2=self.C2(self.D1(R1))
        R3=self.C3(self.D2(R2))
        R4=self.C4(self.D3(R3))
        Y1=self.C5(self.D4(R4))
        O1=self.C6(self.U1(Y1,R4))
        O2=self.C7(self.U2(O1,R3))
        O3=self.C8(self.U3(O2,R2))
        O4=self.C9(self.U4(O3,R1))
        # return self.Th(self.pre(O4))
        return self.pre(O4)

if __name__ == '__main__':
    x=torch.randn(2,3,256,256)
    net=net()
    print(net(x).shape)
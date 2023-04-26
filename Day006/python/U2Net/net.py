import torch
import torch.nn as nn
import torch.nn.functional as F

class CBM(nn.Module):
    def __init__(self,in_chl=3,out_chl=3,dirate=1):
        super(CBM, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_chl,in_chl//2,(3,3),(1,1),(1,1),padding_mode="reflect",bias=False),
            nn.BatchNorm2d(in_chl//2),
            nn.Dropout2d(0.3),
            nn.Mish(inplace=True),
            nn.Conv2d(in_chl//2, in_chl//2, (3, 3), (1, 1), padding=(1 * dirate, 1 * dirate),
                      dilation=(1 * dirate, 1 * dirate),bias=False),
            nn.BatchNorm2d(in_chl//2),
            nn.Dropout2d(0.4),
            nn.Mish(inplace=True),
            nn.Conv2d(in_chl//2, out_chl, (3, 3), (1, 1), (1, 1), padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_chl),
            nn.Dropout2d(0.3),
            nn.Mish(inplace=True)
        )

    def forward(self,x):
        return self.layer(x)

def _upsample_like(src,tar):
    src=F.interpolate(src,size=tar.shape[2:],mode="nearest")
    return src

class RSU_7(nn.Module):
    def __init__(self,in_chl=3,mid_chl=12,out_chl=3):
        super(RSU_7, self).__init__()
        self.rebnconvin=CBM(in_chl,out_chl,dirate=1)
        self.rebnconv1=CBM(out_chl,mid_chl,dirate=1)
        self.pool1=nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2=CBM(mid_chl,mid_chl,dirate=1)
        self.pool2=nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3=CBM(mid_chl, mid_chl,dirate=1)
        self.pool3=nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4=CBM(mid_chl, mid_chl,dirate=1)
        self.pool4=nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5=CBM(mid_chl, mid_chl,dirate=1)
        self.pool5=nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6=CBM(mid_chl, mid_chl, dirate=1)

        self.rebnconv7=CBM(mid_chl, mid_chl, dirate=2)

        self.rebnconv6d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv5d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv4d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv3d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv2d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv1d=CBM(mid_chl*2, out_chl, dirate=1)

    def forward(self,x):
        hx=x
        hxin=self.rebnconvin(hx)

        hx1=self.rebnconv1(hxin)
        hx=self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6=self.rebnconv6(hx)

        hx7=self.rebnconv7(hx6)

        hx6d=self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup=_upsample_like(hx6d,hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d=self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d+hxin

class RSU_6(nn.Module):
    def __init__(self,in_chl=3,mid_chl=12,out_chl=3):
        super(RSU_6, self).__init__()
        self.rebnconvin=CBM(in_chl,out_chl,dirate=1)
        self.rebnconv1=CBM(out_chl,mid_chl,dirate=1)
        self.pool1=nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2=CBM(mid_chl,mid_chl,dirate=1)
        self.pool2=nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3=CBM(mid_chl, mid_chl,dirate=1)
        self.pool3=nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4=CBM(mid_chl, mid_chl,dirate=1)
        self.pool4=nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5=CBM(mid_chl, mid_chl, dirate=1)

        self.rebnconv6=CBM(mid_chl, mid_chl, dirate=2)

        self.rebnconv5d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv4d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv3d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv2d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv1d=CBM(mid_chl*2, out_chl, dirate=1)

    def forward(self,x):
        hx=x
        hxin=self.rebnconvin(hx)

        hx1=self.rebnconv1(hxin)
        hx=self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6=self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d=self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d+hxin

class RSU_5(nn.Module):
    def __init__(self,in_chl=3,mid_chl=12,out_chl=3):
        super(RSU_5, self).__init__()
        self.rebnconvin=CBM(in_chl,out_chl,dirate=1)
        self.rebnconv1=CBM(out_chl,mid_chl,dirate=1)
        self.pool1=nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2=CBM(mid_chl,mid_chl,dirate=1)
        self.pool2=nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3=CBM(mid_chl, mid_chl,dirate=1)
        self.pool3=nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4=CBM(mid_chl, mid_chl, dirate=1)

        self.rebnconv5=CBM(mid_chl, mid_chl, dirate=2)

        self.rebnconv4d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv3d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv2d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv1d=CBM(mid_chl*2, out_chl, dirate=1)

    def forward(self,x):
        hx=x
        hxin=self.rebnconvin(hx)

        hx1=self.rebnconv1(hxin)
        hx=self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5=self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d=self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d+hxin

class RSU_4(nn.Module):
    def __init__(self,in_chl=3,mid_chl=12,out_chl=3):
        super(RSU_4, self).__init__()
        self.rebnconvin=CBM(in_chl,out_chl,dirate=1)
        self.rebnconv1=CBM(out_chl,mid_chl,dirate=1)
        self.pool1=nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2=CBM(mid_chl,mid_chl,dirate=1)
        self.pool2=nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3=CBM(mid_chl, mid_chl, dirate=1)

        self.rebnconv4=CBM(mid_chl, mid_chl, dirate=2)

        self.rebnconv3d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv2d=CBM(mid_chl*2, mid_chl, dirate=1)
        self.rebnconv1d=CBM(mid_chl*2, out_chl, dirate=1)

    def forward(self,x):
        hx=x
        hxin=self.rebnconvin(hx)

        hx1=self.rebnconv1(hxin)
        hx=self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4=self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d=self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d+hxin

class RSU(nn.Module):
    def __init__(self,in_chl=3,mid_chl=12,out_chl=3):
        super(RSU, self).__init__()
        self.rebnconvin=CBM(in_chl,out_chl,dirate=1)

        self.rebnconv1 = CBM(out_chl, mid_chl, dirate=1)
        self.rebnconv2 = CBM(mid_chl, mid_chl, dirate=2)
        self.rebnconv3 = CBM(mid_chl, mid_chl, dirate=4)

        self.rebnconv4 = CBM(mid_chl, mid_chl, dirate=8)

        self.rebnconv3d = CBM(mid_chl*2, mid_chl, dirate=4)
        self.rebnconv2d = CBM(mid_chl*2, mid_chl, dirate=2)
        self.rebnconv1d = CBM(mid_chl*2, out_chl, dirate=1)

    def forward(self,x):
        hx=x

        hxin=self.rebnconvin(hx)

        hx1=self.rebnconv1(hxin)
        hx2=self.rebnconv2(hx1)
        hx3=self.rebnconv3(hx2)

        hx4=self.rebnconv4(hx3)

        hx3d=self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d=self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d=self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d+hxin


class U2NET(nn.Module):
    def __init__(self,in_chl=3,out_chl=3):
        super(U2NET, self).__init__()
        self.stage1=RSU_7(in_chl,32,64)
        self.pool12=nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2=RSU_6(64,32,128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3=RSU_5(128,64,256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4=RSU_4(256,128,512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5=RSU(512,256,512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU(512, 256, 512)

        #decoder
        self.stage5d=RSU(1024,256,512)
        self.stage4d=RSU_4(1024,128,256)
        self.stage3d=RSU_5(512,64,128)
        self.stage2d=RSU_6(256,32,64)
        self.stage1d=RSU_7(128,16,64)
        #output
        self.side1=nn.Conv2d(64,out_chl,(3,3),padding=1)
        self.side2=nn.Conv2d(64,out_chl,(3,3),padding=1)
        self.side3=nn.Conv2d(128,out_chl,(3,3),padding=1)
        self.side4=nn.Conv2d(256,out_chl,(3,3),padding=1)
        self.side5=nn.Conv2d(512,out_chl,(3,3),padding=1)
        self.side6=nn.Conv2d(512,out_chl,(3,3),padding=1)

        self.outconv=nn.Conv2d(18,out_chl,(1,1))

    def forward(self,x):
        hx1=self.stage1(x)
        hx=self.pool12(hx1)

        hx2=self.stage2(hx)
        hx=self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5up, hx4), 1))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4up, hx3), 1))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3up, hx2), 1))
        hx2up = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2up, hx1), 1))

        #out
        d1=self.side1(hx1d)

        d2=self.side2(hx2d)
        d2=_upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0=self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        # return F.sigmoid(d0),F.sigmoid(d1),F.sigmoid(d2),F.sigmoid(d3),F.sigmoid(d4),F.sigmoid(d5),F.sigmoid(d6)
        return d0,d1,d2,d3,d4,d5,d6

class U2NET_S(nn.Module):
    def __init__(self,in_chl=3,out_chl=3):
        super(U2NET_S, self).__init__()
        self.stage1=RSU_7(in_chl,16,64)
        self.pool12=nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2=RSU_6(64,16,64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3=RSU_5(64,16,64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4=RSU_4(64,16,64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5=RSU(64,16,64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU(64,16,64)

        #decoder
        self.stage5d=RSU(128,16,64)
        self.stage4d=RSU_4(128,16,64)
        self.stage3d=RSU_5(128,16,64)
        self.stage2d=RSU_6(128,16,64)
        self.stage1d=RSU_7(128,16,64)
        #output
        self.side1=nn.Conv2d(64,out_chl,(3,3),padding=1)
        self.side2=nn.Conv2d(64,out_chl,(3,3),padding=1)
        self.side3=nn.Conv2d(64,out_chl,(3,3),padding=1)
        self.side4=nn.Conv2d(64,out_chl,(3,3),padding=1)
        self.side5=nn.Conv2d(64,out_chl,(3,3),padding=1)
        self.side6=nn.Conv2d(64,out_chl,(3,3),padding=1)

        self.outconv=nn.Conv2d(18,out_chl,(1,1))

    def forward(self,x):
        hx1=self.stage1(x)
        hx=self.pool12(hx1)

        hx2=self.stage2(hx)
        hx=self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5up = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5up, hx4), 1))
        hx4up = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4up, hx3), 1))
        hx3up = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3up, hx2), 1))
        hx2up = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2up, hx1), 1))

        #out
        d1=self.side1(hx1d)

        d2=self.side2(hx2d)
        d2=_upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0=self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        # return F.sigmoid(d0),F.sigmoid(d1),F.sigmoid(d2),F.sigmoid(d3),F.sigmoid(d4),F.sigmoid(d5),F.sigmoid(d6)
        return d0, d1, d2, d3, d4, d5, d6

if __name__ == '__main__':
    net=U2NET()
    x=torch.randn(1,3,224,224)
    y=net(x)
    print(y[0].shape)
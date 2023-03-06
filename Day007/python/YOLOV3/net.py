import torch
from torch import nn
import torch.nn.functional as F

class Upsamplelasyer(nn.Module):
    def __init__(self):
        super(Upsamplelasyer, self).__init__()

    def forward(self,x):
        return F.interpolate(x,scale_factor=2,mode='nearest')

class ConvolutionalLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(ConvolutionalLayer, self).__init__()
        self.sub_module=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self,x):
        return self.sub_module(x)

class ResidualLayer(nn.Module):
    def __init__(self,in_channels):
        super(ResidualLayer, self).__init__()
        self.sub_module=nn.Sequential(
            ConvolutionalLayer(in_channels,in_channels//2,1,1,0),
            ConvolutionalLayer(in_channels//2,in_channels,3,1,1)
        )

    def forward(self,x):
        return x+self.sub_module(x)

class DownsamplingLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownsamplingLayer, self).__init__()
        self.sub_module=nn.Sequential(
            ConvolutionalLayer(in_channels,out_channels,3,2,1)
        )

    def forward(self,x):
        return self.sub_module(x)

class ConvolutionalSet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvolutionalSet, self).__init__()
        self.sub_module=nn.Sequential(
            ConvolutionalLayer(in_channels,out_channels,1,1,0),
            ConvolutionalLayer(out_channels,in_channels,3,1,1),

            ConvolutionalLayer(in_channels,out_channels,1,1,0),
            ConvolutionalLayer(out_channels,in_channels,3,1,1),

            ConvolutionalLayer(in_channels,out_channels,1,1,0)
        )

    def forward(self,x):
        return self.sub_module(x)

class Yolov3(nn.Module):
    def __init__(self):
        super(Yolov3, self).__init__()

        self.trunk_52=nn.Sequential(
            ConvolutionalLayer(3,32,3,1,1),
            ConvolutionalLayer(32,64,3,2,1),

            ResidualLayer(64),
            DownsamplingLayer(64,128),

            ResidualLayer(128),
            ResidualLayer(128),
            DownsamplingLayer(128,256),

            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256)
        )

        self.trunk_26=nn.Sequential(
            DownsamplingLayer(256,512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512)
        )

        self.trunk_13=nn.Sequential(
            DownsamplingLayer(512,1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )

        self.convset_13=nn.Sequential(
            ConvolutionalSet(1024,512)
        )

        self.detection_13=nn.Sequential(
            ConvolutionalLayer(512,1024,3,1,1),
            nn.Conv2d(1024,45,(1,1),(1,1),0)
        )

        self.up_26=nn.Sequential(
            ConvolutionalLayer(512,256,3,1,1),
            Upsamplelasyer()
        )

        self.convset_26=nn.Sequential(
            ConvolutionalSet(768,256)
        )

        self.detection_26=nn.Sequential(
            ConvolutionalLayer(256,512,3,1,1),
            nn.Conv2d(512,45,(1,1),(1,1),0)
        )

        self.up_52=nn.Sequential(
            ConvolutionalLayer(256,128,3,1,1),
            Upsamplelasyer()
        )

        self.convset_52=nn.Sequential(
            ConvolutionalSet(384,128)
        )

        self.detection_52=nn.Sequential(
            ConvolutionalLayer(128,256,3,1,1),
            nn.Conv2d(256,45,(1,1),(1,1),0)
        )

    def forward(self,x):

        return 0

if __name__ == '__main__':
    yolo=Yolov3()
    x=torch.randn(1,3,416,416)
    y=yolo(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
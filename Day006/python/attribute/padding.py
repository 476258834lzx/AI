#填充
import torch
from torch import nn

def padding():
    img=torch.randn(1,1,9,9)
    conv=nn.Conv2d(1,1,(3,3),(1,1),(1,1))#padding上下加1,padding左右加1
    out=conv(img)
    print(out.shape)
    weight=conv.weight
    print(weight)
    print(weight.shape)#卷积核个数、通道数、卷积核宽高
    padding=conv.padding
    print(padding)
    bias=conv.bias
    print(bias)
    #空洞卷积
    conv1=nn.Conv2d(1,1,(3,3),(1,1),padding=(0,0),dilation=(2,2))#一个像素点和下一个像素点之间的距离
padding()
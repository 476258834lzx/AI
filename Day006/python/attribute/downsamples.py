#下采样
import torch
from torch import nn

def pooling():
    conv_img=torch.randn(1,1,9,9)
    max_pooling=nn.MaxPool2d(2)
    avg_pooling=nn.AvgPool2d(2)
    #自适应池化,池化的窗口和步长根据输入图像大小自动计算得到
    adapt_max_pooling=nn.AdaptiveMaxPool2d((3,5))#参数:输出的特征图形状
    adapt_avg_pooling=nn.AdaptiveAvgPool2d((3,5))#参数:输出的特征图形状
    out=max_pooling(conv_img)
    print(out.shape)
    out1=adapt_max_pooling(conv_img)
    print(out1.shape)
pooling()
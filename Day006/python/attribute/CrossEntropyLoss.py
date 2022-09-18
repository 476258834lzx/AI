#交叉熵
from torch import nn
import torch

class CE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()

    def forward(self,y,p):
        out=torch.mean(-(y*torch.log(p)))
        return out
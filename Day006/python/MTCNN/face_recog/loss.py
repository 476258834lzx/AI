import torch
from torch import nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    def __init__(self,cls_num,feature_num):
        super(CenterLoss, self).__init__()
        self.cls_num=cls_num
        self.center=nn.Parameter(torch.randn(cls_num,feature_num))

    def forward(self,datas,labels):
        center_exp = self.center.index_select(dim=0, index=labels.long())#labels为onehot编码
        count = torch.histc(labels.float(), bins=self.cls_num, min=0, max=self.cls_num-1)
        count_exp = count.index_select(dim=0, index=labels.long())
        center_loss = torch.sum(torch.div(torch.sqrt(torch.sum(torch.pow(datas - center_exp, 2), dim=1)), count_exp))
        return center_loss

class ArcsoftmaxLoss(nn.Module):
    def __init__(self,feature_num,cls_num):
        super(ArcsoftmaxLoss, self).__init__()
        self.w=nn.Parameter(torch.randn(feature_num,cls_num))

    def forward(self,x,s=1,m=0.2):
        x_normal=F.normalize(x,dim=1)
        w_normal=F.normalize(self.w,dim=0)
        cos=torch.matmul(x_normal,w_normal)/10#防止arc函数梯度爆炸
        a=torch.arccos(cos)
        arcloss=torch.exp(s*torch.cos(a+m)*10)/(torch.exp(s*torch.cos(a+m)*10)+torch.sum(torch.exp(s*cos*10),dim=1,keepdim=True)-torch.exp(s*cos*10))
        return torch.log(arcloss)#配合NLLLOSS使用,该Api默认没有使用log
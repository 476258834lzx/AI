import torch
from torch import nn

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

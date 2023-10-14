# import torch
# from torch import nn
#
# transformer_model=nn.Transformer()
# src=torch.randn((10,32,512))
# dst=torch.randn((10,32,512))
# out=transformer_model(src,dst)
# print(out.shape)


#公式实现
import torch
from torch import nn
from torch.nn.functional import softmax
class Attention(nn.Module):
    #NCHW-NHWC->NV,单batch公式
    def __init__(self,head_num,feature_num,clamp_feature):
        super(Attention, self).__init__()
        self.head_num=head_num
        self.feature_num=feature_num
        self.feature_num=feature_num
        self.clamp_feature=clamp_feature
        self.fc=nn.Linear(feature_num,feature_num*head_num)
        self.fc1=nn.Linear(clamp_feature,clamp_feature*head_num)
        self.fc2=nn.Linear(clamp_feature*head_num,clamp_feature)
    def forward(self,Q,K,V):
        QS=torch.stack(self.fc(Q).chunk(self.head_num,dim=-1))#NV->SNV,S为头数
        KS=torch.stack(self.fc(K).chunk(self.head_num,dim=-1))
        VS=torch.stack(self.fc1(V).chunk(self.head_num,dim=-1))
        dk = self.feature_num ** 0.5

        # A=softmax((QS@KS.permute(0,2,1))/dk,dim=-1)@VS#SNV*SVN=SNN@VS(SNV),单batch公式
        A = softmax((QS @ KS.transpose(-1,-2)) / dk, dim=-1) @ VS#HNSV*HNVS=HNSS*HNSV=HNSV
        AA=torch.cat(A.permute(1,2,0,3).chunk(self.head_num,dim=2),dim=-1).squeeze()#NSV
        return self.fc2(AA)


    def single(self,Q,K,V):
        dk=self.feature_num**0.5
        return softmax((Q@K.transpose(-1,-2))/dk,dim=-1)@V

if __name__ == '__main__':
    q=torch.randn(2,3,5)
    k=torch.randn(2,7,5)
    v=torch.randn(2,7,6)#V求完特征相关度对特征向量进行截取
    att=Attention(1,5,6)
    y=att.single(q,k,v)
    print(y.shape)

    att1=Attention(3,5,6)
    y1=att1(q,k,v)
    print(y1.shape)
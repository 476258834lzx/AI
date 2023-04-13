import torch
from torch import nn

transformer_model=nn.Transformer()
src=torch.randn((10,32,512))
dst=torch.randn((10,32,512))
out=transformer_model(src,dst)
print(out.shape)


#diy
# import torch
# from torch import nn
# from torch.nn.functional import softmax
# class Attention(nn.Module):
#     #NCHW-NHWC->NV
#     def __init__(self,head_num,feature_num):
#         super(Attention, self).__init__()
#         self.head_num=head_num
#         self.feature_num=feature_num
#         self.fc=nn.Linear(feature_num,feature_num*head_num)
#         self.fc2=nn.Linear(feature_num*head_num,feature_num)
#     def forward(self,Q,K,V):
#         QS=torch.stack(self.fc(Q).chunk(self.head_num,dim=1))#NV->SNV
#         KS=torch.stack(self.fc(K).chunk(self.head_num,dim=1))
#         VS=torch.stack(self.fc(V).chunk(self.head_num,dim=1))
#
#         dk = Q.shape[1] ** 0.5
#         A=softmax((QS@KS.permute(0,2,1))/dk,dim=-1)@VS#SNV*SVN=SNN@VS(SNV)
#         AA=A.permute(1,0,2).reshape(Q.shape[0],-1)#NSV
#         return self.fc2(AA)
#
#
#     def single(self,Q,K,V):
#         dk=Q.shape[1]**0.5
#         return softmax((Q@K.T)/dk,dim=-1)@V
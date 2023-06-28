#构建大模型
#构建小模型
#构建数据
#KL散度损失+交叉熵损失

import os
import torch
from torch import nn as nn
from torch.utils.data import TensorDataset,DataLoader,SequentialSampler
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class model(nn.Module):
    def __init__(self,input_dim,hiddien_dim,output_dim):
        super(model, self).__init__()
        self.layer1=nn.LSTM(input_dim,hiddien_dim,output_dim)
        self.layer2=nn.Linear(hiddien_dim,output_dim)
    def forward(self,inputs):
        layer1_output,layer1_hidden=self.layer1(inputs)
        layer2_output=self.layer2(layer1_output)
        layer2_output=layer2_output[:,-1,:]
        return layer2_output

#创建小模型
model_student=model(input_dim=2,hiddien_dim=8,output_dim=4)
model_student=model_student.to(DEVICE)
#创建大模型
model_teacher=model(input_dim=2,hiddien_dim=16,output_dim=4)
model_teacher.load_state_dict(torch.load(os.path.join("weights","last.pt")))
model_teacher=model_teacher.to(DEVICE)
#加载数据
inputs=torch.randn(4,6,2)
true_label=torch.tensor([0,1,0,0])

dataset=TensorDataset(inputs,true_label)
sampler=SequentialSampler(inputs)
dataloader=DataLoader(dataset,sampler=sampler,batch_size=2)

loss_func=nn.CrossEntropyLoss()
#kl损失
klloss_func=nn.KLDivLoss()
optmizier=torch.optim.SGD(model_student.parameters(),lr=0.1,momentum=0.9)

for step,(x,y) in enumerate(dataloader):
    x,y=x.to(DEVICE),y.to(DEVICE)
    out_student=model_student(x)
    out_teacher=model_teacher(x)

    loss_hard=loss_func(out_student,y)
    loss_kl=klloss_func(out_student,out_teacher)
    loss=0.5*loss_kl+0.5*loss_hard

    optmizier.zero_grad()
    loss.backward()
    optmizier.step()

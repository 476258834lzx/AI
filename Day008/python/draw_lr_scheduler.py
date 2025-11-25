import torch
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import torch.nn as nn
from torchvision.models import densenet121
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
model=densenet121()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
mode="cosineAnn"
if mode=="cosineAnn":
    scheduler=CosineAnnealingLR(optimizer,T_max=10,eta_min=0)
elif mode=="cosineAnnWarm":
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)

plt.figure()
max_epoch=50
iters=200
cur_lr_list=[]

for epoch in range(max_epoch):
    for batch in range(iters):
        optimizer.step()
    scheduler.step()
    cur_lr=optimizer.param_groups[-1]["lr"]
    cur_lr_list.append(cur_lr)
x_list=list(range(len(cur_lr_list)))
plt.plot(x_list,cur_lr_list)
plt.show()
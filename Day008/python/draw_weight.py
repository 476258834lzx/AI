import torch
from net import PNet
from torch.utils.tensorboard import SummaryWriter
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'
pnet=PNet()
pnet.load_state_dict(torch.load("pnet.pt"))
# print(pnet.pre_layer[0].weight)
print(pnet)
summarywriter=SummaryWriter("weight_log")
layer1=pnet.pre_layer[0].weight
layer2=pnet.pre_layer[3].weight
layer3=pnet.pre_layer[5].weight
summarywriter.add_histogram("layer1",layer1)#数据名，数据，步长
summarywriter.add_histogram("layer2",layer2)
summarywriter.add_histogram("layer3",layer3)
print("666")
time.sleep(2)
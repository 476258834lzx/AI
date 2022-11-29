import torch.nn as nn
from torch.nn.functional import one_hot
import torch
import torch.utils.data as data
import torchvision.datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from mydata import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataset=MINISTDataset(r"D:\Relearn\Day005\python\diy_num_project\data",True)
train_dataloader=data.DataLoader(train_dataset,batch_size=512,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
        self.out=nn.Linear(2,10)

    def forward(self,x):
        fc1=self.layer(x)
        fc2=self.out(fc1)

        return fc1,fc2
        #N2    N
def draw(feat,labels,epoch):
    plt.ion()
    color=["#CC99CC","#9933FF","#0099FF","#FF3300","#33FF99","#336633","#FF99FF","#CCCCFF","#FFCC99","#66FFCC"]
    plt.cla()
    for i in range(10):
        plt.plot(feat[labels==i,0],feat[labels==i,1],".",c=color[i])
        plt.legend(["0","1","2","3","4","5","6","7","8","9"],loc="upper right")
        plt.title("epoch=%d"%epoch)
        plt.savefig("feature/epoch=%d.jpg"%epoch)
        plt.draw()
        plt.pause(0.01)

if __name__ == '__main__':
    net=Net().to(device)
    loss_func=nn.MSELoss()
    opt=torch.optim.SGD(net.parameters(),lr=0.001)

    epoch=0
    while True:
        feat_list=[]
        label_list=[]
        for i,(x,y) in enumerate(train_dataloader):
            x=x.reshape(-1,784).to(device)
            target=y.to(device).float()

            feat,out=net(x)
            loss=loss_func(out,target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            feat_list.append(feat)
            label_list.append(y.argmax(dim=1))

            print(loss.item())
        feat=torch.cat(feat_list,dim=0)
        label=torch.cat(label_list,dim=0)

        draw(feat.detach().cpu().numpy(),label.detach().cpu().numpy(),epoch)
        epoch+=1

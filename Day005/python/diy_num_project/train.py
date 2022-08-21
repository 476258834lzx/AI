import torch
import os
from torch import nn,optim
from net import *
from data import MINISTDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
os.environ['KMP_DUPLICATE_LIB_OK']='True'

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Train:
    def __init__(self,root):
        #初始化
        self.summaryWriter=SummaryWriter("logs")

        #加载训练数据
        self.train_dataset=MINISTDataset(root,True)
        self.train_dataloader=DataLoader(self.train_dataset,100,True,drop_last=True,num_workers=8)

        #加载测试数据
        self.test_dataset=MINISTDataset(root,False)
        self.test_dataloader = DataLoader(self.test_dataset, 100, True, drop_last=True,num_workers=8)

        #加载模型
        self.net=NetV3().to(DEVICE)

        # 加载预训练权重
        if os.path.exists("params/0.pt"):
            self.net.load_state_dict(torch.load("params/0.pt"))

        #优化算法
        self.opt=optim.Adam(self.net.parameters())

        #损失函数
        self.loss_function=nn.MSELoss()

    #训练过程
    def __call__(self):
        for epoch in range(10000):
            train_loss=0
            test_loss=0
            acc=0
            for i,(imgs,tags) in enumerate(self.train_dataloader):
                imgs, tags=imgs.to(DEVICE),tags.to(DEVICE)
                #开启训练模式，网络
                self.net.train()
                #100*784(batchsize*784)
                y=self.net.forward(imgs)
                #100*10
                train_loss=torch.mean((tags-y)**2)#对应位置相减
                # train_loss=self.loss_function(y,tags)

                self.opt.zero_grad()
                train_loss.backward()
                self.opt.step()

            #验证
            for i,(imgs,tags)in enumerate(self.test_dataloader):
                imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)
                #开启测试模式，网络
                self.net.eval()
                test_y=self.net(imgs)
                test_loss=torch.mean((tags-test_y)**2)
                #one-hot编码转标签
                # print(torch.argmax(test_y,dim=1))
                # print(torch.argmax(tags,dim=1))
                predict_tags=torch.argmax(test_y,dim=1)
                label_tags=torch.argmax(tags,dim=1)
                #准确度，预测正确个数/标签总数   不同于PR计算
                acc=(predict_tags == label_tags).sum().item()/len(tags)#相同为True不同为False的掩码矩阵
            # 收集损失
            # self.summaryWriter.add_scalar("train_loss", loss, epoch)
            self.summaryWriter.add_scalars("loss", {"train_loss":train_loss,"test_loss":test_loss},epoch)
            if epoch%10==0:
                print("train_loss:",train_loss.item(),"test_loss:",test_loss.item(),"accuracy:",acc)
            if epoch%50==0:
                torch.save(self.net.state_dict(),f"params/{epoch}.pth")
if __name__ == '__main__':
    train=Train("data")
    train()


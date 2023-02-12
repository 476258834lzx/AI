from face_dataset import Face_dataset
from torch import optim
from torch.utils.data import DataLoader
from face_net import *
from loss import *
from torch.utils.tensorboard import SummaryWriter
import os

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Train:
    def __init__(self,img_dir,weights_dir):
        #初始化
        self.summaryWriter=SummaryWriter("logs")

        #加载训练数据
        self.train_dataset=Face_dataset(img_dir,True)
        self.train_dataloader=DataLoader(self.train_dataset,32,True,drop_last=True,num_workers=8)

        #加载测试数据
        self.val_dataset=Face_dataset(img_dir,False)
        self.val_dataloader = DataLoader(self.val_dataset, 16, True, drop_last=True,num_workers=8)

        #加载模型
        self.net=FaceNet().to(DEVICE)

        # 加载预训练权重
        self.weights_dir=weights_dir
        if os.path.exists(os.path.join(self.weights_dir,"last.pt")):
            self.net.load_state_dict(torch.load(os.path.join(self.weights_dir,"last.pt")))

        #优化算法
        # self.opt=optim.Adam(self.net.parameters())
        self.opt=optim.SGD(self.net.parameters(),lr=0.0001,momentum=0.9)
        self.exp_lr_scheduler=optim.lr_scheduler.StepLR(self.opt,step_size=10,gamma=0.5)

        #损失函数
        self.loss_function=nn.NLLLoss()

    def __call__(self):
        last_accuracy = 0
        for epoch in range(10000):
            sum_train_loss=0
            sum_val_loss=0
            sum_accuracy=0
            train_account = len(self.train_dataloader)
            val_account = len(self.val_dataloader)
            #训练
            for i,(imgs,tags) in enumerate(self.train_dataloader):
                imgs,tags=imgs.to(DEVICE),tags.to(DEVICE)
                self.net.train()
                feature,y=self.net(imgs)
                train_loss=self.loss_function(y,tags)

                self.opt.zero_grad()
                train_loss.backward()
                self.opt.step()
                sum_train_loss += train_loss.detach().cpu().item()
            #测试
            for i,(imgs,tags) in enumerate(self.val_dataloader):
                imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)
                self.net.eval()
                feature,y=self.net(imgs)
                val_loss=self.loss_function(y,tags)
                sum_val_loss+=val_loss.detach().cpu().item()
                predict_tags=torch.argmax(y,dim=1)
                accuracy=(predict_tags==tags).sum().item()/len(tags)
                sum_accuracy+=accuracy

            if sum_accuracy/val_account > last_accuracy:
                last_accuracy = sum_accuracy/val_account
                torch.save(self.net.state_dict(), f"{self.weights_dir}/best.pt")
            print("train_loss:",sum_train_loss/train_account,"val_loss:",sum_val_loss/val_account,"accuracy:",sum_accuracy/val_account)
            self.summaryWriter.add_scalars("loss",{"train_loss":sum_train_loss/train_account,"val_loss":sum_val_loss/val_account},epoch)
            torch.save(self.net.state_dict(), f"{self.weights_dir}/last.pt")
            self.exp_lr_scheduler.step()

if __name__ == '__main__':
    train=Train("img","params")
    train()
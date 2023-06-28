import torch.nn
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from YOLOV3.dataset import Mydata
from net import *
from torch import optim

os.environ['KMP_DUPLICATE_LIB_OK']='True'
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None,ignore_index=255,is_BCE=False):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        if is_BCE:
            self.ce_fn = nn.BCEWithLogitsLoss(weight=self.weight, reduction='none')
        else:
            self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, reduction='none')

    def forward(self, input, target):
        # target=torch.maximum(target,torch.tensor([1e-6]).long().cuda())
        logpt = -self.ce_fn(input, target)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        loss_=loss.mean()
        return loss_

def loss_fun(output,target):
    output=output.permute(0,2,3,1)#N,30,13,13->N,13,13,30
    output=output.reshape(output.size(0),output.size(1),output.size(2),3,-1)#N,13,13,30->N,13,13,3,10
    mask_obj=target[...,0]>0
    #mask_noobj=target[...,0]==0#正负样本损失,YOLOV45加focalloss在此
    conf_lossfunc=FocalLoss(is_BCE=True)
    landmark_lossfunc=nn.MSELoss()
    cls_lossfunc=FocalLoss()

    loss_conf=conf_lossfunc(output[...,0],target[...,0])
    if len(target[mask_obj])==0:
        loss_landmark=0
        loss_cls=0
    else:
        loss_landmark=landmark_lossfunc(output[mask_obj][...,1:5],target[mask_obj][...,1:5])#没有目标该跳过
        loss_cls=cls_lossfunc(output[mask_obj][...,5:10],target[mask_obj][...,5].long())
    loss=loss_conf+loss_landmark+loss_cls

    return loss

def evaluate(output,target):
    output = output.permute(0, 2, 3, 1)  # N,18,13,13->N,13,13,18
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)  # N,13,13,18->N,13,13,3,6
    mask_obj = target[..., 0] > 0
    predict_tags = torch.argmax(output[mask_obj][..., 5:10], dim=1)
    ground_tags = target[mask_obj][..., 5]
    accuracy = (predict_tags == ground_tags).sum().float().mean().item()

    return accuracy

class Train:
    def __init__(self,img_dir,weights_dir):
        #初始化
        self.summaryWriter=SummaryWriter("logs")

        #加载训练数据
        self.train_dataset=Mydata(img_dir,True)
        self.train_dataloader=DataLoader(self.train_dataset,32,True,drop_last=True,num_workers=8)

        #加载测试数据
        self.val_dataset=Mydata(img_dir,False)
        self.val_dataloader = DataLoader(self.val_dataset, 16, True, drop_last=True,num_workers=8)

        #加载模型
        self.net=Yolov3().to(DEVICE)

        # 加载预训练权重
        self.weights_dir=weights_dir
        if os.path.exists(os.path.join(self.weights_dir,"last.pt")):
            self.net.load_state_dict(torch.load(os.path.join(self.weights_dir,"last.pt")))

        #优化算法
        self.opt=optim.Adam(self.net.parameters())
        # self.opt=optim.SGD(self.net.parameters(),lr=0.01,momentum=0.9)
        # self.exp_lr_scheduler=optim.lr_scheduler.StepLR(self.opt,step_size=150,gamma=0.5)

    def __call__(self):
        last_accuracy = 0
        for epoch in range(10000):
            sum_train_loss=0
            sum_val_loss=0
            sum_accuracy=0
            train_account = len(self.train_dataloader)
            val_account = len(self.val_dataloader)
            #训练
            self.net.train()
            for target_13,target_26,target_52,img_data in self.train_dataloader:
                target_13, target_26, target_52, img_data=target_13.to(DEVICE),target_26.to(DEVICE),target_52.to(DEVICE),img_data.to(DEVICE)

                output_13, output_26, output_52 = self.net(img_data)
                loss_13 = loss_fun(output_13, target_13)
                loss_26 = loss_fun(output_26, target_26)
                loss_52 = loss_fun(output_52, target_52)

                train_loss=loss_13+loss_26+loss_52

                self.opt.zero_grad()
                train_loss.backward()
                self.opt.step()
                sum_train_loss += train_loss.detach().cpu().item()
            #测试
            self.net.eval()
            for target_13,target_26,target_52,img_data in self.val_dataloader:
                target_13, target_26, target_52, img_data=target_13.to(DEVICE),target_26.to(DEVICE),target_52.to(DEVICE),img_data.to(DEVICE)

                output_13, output_26, output_52 = self.net(img_data)
                loss_13 = loss_fun(output_13, target_13)
                loss_26 = loss_fun(output_26, target_26)
                loss_52 = loss_fun(output_52, target_52)

                val_loss = loss_13 + loss_26 + loss_52
                sum_val_loss+=val_loss.detach().cpu().item()

                accuracy_13=evaluate(output_13,target_13)
                accuracy_26=evaluate(output_26,target_26)
                accuracy_52=evaluate(output_52,target_52)
                accuracy=accuracy_13+accuracy_26+accuracy_52

                sum_accuracy+=accuracy

            if sum_accuracy/val_account > last_accuracy:
                last_accuracy = sum_accuracy/val_account
                torch.save(self.net.state_dict(), f"{self.weights_dir}/best.pt")
            print("train_loss:",sum_train_loss/train_account,"val_loss:",sum_val_loss/val_account,"accuracy:",sum_accuracy/val_account)
            self.summaryWriter.add_scalars("loss",{"train_loss":sum_train_loss/train_account,"val_loss":sum_val_loss/val_account},epoch)
            torch.save(self.net.state_dict(), f"{self.weights_dir}/last.pt")
            # self.exp_lr_scheduler.step()

if __name__ == '__main__':
    train = Train("data/fisheye_parking","weights")
    train()
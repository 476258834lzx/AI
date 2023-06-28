import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from data import FaceDataset

class Trainer:
    def __init__(self,net,size,save_path,dataset_path,isCuda=True):
        self.net=net()
        self.save_path=save_path
        self.dataset_path=dataset_path
        self.isCuda=isCuda
        self.size=size

        if self.isCuda:
            self.net.cuda()

        self.cls_loss_fn=nn.BCELoss()
        self.offset_loss_fn=nn.MSELoss()
        self.landmark_loss_fn=nn.MSELoss()

        self.optimizer=optim.Adam(self.net.parameters())

        if os.path.exists(self.save_path):
            self.net.load_state_dict(torch.load(self.save_path))

    def train(self):
        faceDataset=FaceDataset(self.dataset_path,self.size)
        dataloader=DataLoader(faceDataset,batch_size=512,shuffle=True,drop_last=True)
        epoch=0
        while True:
            sum_loss = 0
            sum_cls_loss = 0
            sum_offset_loss = 0
            sum_landmark_loss = 0#取一个轮次中所有batch列表的损失的平均值
            account = len(dataloader)
            for i ,(img_data,cond,offset,landmark) in enumerate(dataloader):#共20000张图，一个batch取512，enumerate中为batch列表
                if self.isCuda:
                    img_data=img_data.cuda()
                    cond=cond.cuda()
                    offset=offset.cuda()
                    landmark=landmark.cuda()

                output_cond,output_offset,output_landmark=self.net(img_data)
                output_cond=output_cond.reshape(-1,1)
                output_offset=output_offset.reshape(-1,4)
                output_landmark=output_landmark.reshape(-1,10)

                cond_mask=torch.lt(cond,2)
                cond_=cond[cond_mask]
                output_cond_=output_cond[cond_mask]
                cls_loss=self.cls_loss_fn(output_cond_,cond_)

                offset_mask=torch.gt(cond,0)
                offset_index=torch.nonzero(offset_mask)[:,0]#取非0元素索引
                offset_=offset[offset_index]
                output_offset_=output_offset[offset_index]
                offset_loss=self.offset_loss_fn(output_offset_,offset_)

                landmark_mask=torch.gt(cond,0)#N*1
                landmark_index = torch.nonzero(offset_mask)[:, 0]
                landmark_ = landmark[landmark_index]
                output_landmark_ = output_landmark[landmark_index]
                landmark_loss=self.landmark_loss_fn(output_landmark_,landmark_)

                loss=cls_loss+offset_loss+landmark_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                sum_loss+=loss.cpu().data.numpy()
                sum_cls_loss+=cls_loss.cpu().data.numpy()
                sum_offset_loss+=offset_loss.cpu().data.numpy()
                sum_landmark_loss+=landmark_loss.cpu().data.numpy()

            print("epoch=", epoch, "loss:", sum_loss / account, "cls_loss:", sum_cls_loss / account, "offset_loss:",
                  sum_offset_loss / account, "landmark_loss:", sum_landmark_loss / account)
            if epoch%10==0:
                torch.save(self.net.state_dict(),self.save_path)
                print("save success!")

            epoch += 1


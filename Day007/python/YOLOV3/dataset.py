import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import cfg
import os
from YOLOV3.utils.general import *

def one_hot(cls_num,v):
    b=np.zeros(cls_num)
    b[v]=1.
    return b

class Mydata(Dataset):
    def __init__(self,data_root,is_Train):
        super(Mydata, self).__init__()
        self.data_root=data_root
        self.train_path=os.path.join(self.data_root,"train.txt")
        self.val_path=os.path.join(self.data_root,"val.txt")

        self.dataset=[]
        if is_Train:
            with open(self.train_path) as f:
                self.dataset=f.readlines()
        else:
            with open(self.val_path) as f:
                self.dataset=f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels={}
        name=self.dataset[index]
        img_path=os.path.join(self.data_root,"images",name[:-1]+".jpg")
        label_path=os.path.join(self.data_root,"labels",name[:-1]+".txt")
        img_data=cv2.imread(img_path)
        ori_height, ori_width = img_data.shape[0:2]
        img_data,top,bottom,left,right,ori_rate=letter_box(img_data,max(cfg.IMG_WIDTH,cfg.IMG_HEIGHT))
        img_data=(img_data/255-0.5).transpose(2,0,1)
        label_data=open(label_path)
        item=0
        for feature_size,anchors in cfg.ANCHORS.items():
            labels[feature_size]=np.zeros((feature_size,feature_size,3,6))
            for line in label_data.readlines():
                strs = line.split()
                cls, _x, _y, _w, _h = strs[0:5]

                x1 = np.float32(_x) * ori_width - np.float32(_w) * ori_width / 2
                y1 = np.float32(_y) * ori_height - np.float32(_h) * ori_height / 2
                x2 = np.float32(_x) * ori_width + np.float32(_w) * ori_width / 2
                y2 = np.float32(_y) * ori_height + np.float32(_h) * ori_height / 2

                new_x1 = x1 * ori_rate + left
                new_y1 = y1 * ori_rate + top
                new_x2 = x2 * ori_rate + left
                new_y2 = y2 * ori_rate + top

                w,h=new_x2-new_x1,new_y2-new_y1
                x = (new_x1 + new_x2) // 2
                y = (new_y1 + new_y2) // 2

                cx_offset,cx_index=np.modf(x*feature_size/max(cfg.IMG_WIDTH,cfg.IMG_HEIGHT))
                cy_offset,cy_index=np.modf(y*feature_size/max(cfg.IMG_WIDTH,cfg.IMG_HEIGHT))

                for i,anchor in enumerate(anchors):
                    anchor_area=cfg.ANCHORS_AREA[feature_size][i]
                    p_w,p_h=w/anchor[0],h/anchor[1]
                    p_area=w*h
                    p_iou=min(p_area,anchor_area)/max(p_area,anchor_area)#YOLOV45更改了iou的计算方式
                    item+=1
                    last_iou=labels[feature_size][int(cx_index),int(cy_index),i,0]
                    if p_iou>=last_iou:
                    #                          H             W                                                 限制值域为-∞到∞
                        labels[feature_size][int(cx_index),int(cy_index),i]=np.array([p_iou,cx_offset,cy_offset,np.log(p_w),np.log(p_h),cls])
        return np.float32(labels[13]),np.float32(labels[26]),np.float32(labels[52]),np.float32(img_data)

if __name__ == '__main__':
    data_path = "data/fisheye_parking"
    dataset = Mydata(data_path,True)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i in data_loader:
        print(i[0].shape)
        print(i[1].shape)
        print(i[2].shape)
        print(i[3].shape)
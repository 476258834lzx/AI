import torch
from torch.utils.data import Dataset,DataLoader
import cv2
import os

class Mydataset(Dataset):
    def __init__(self,path):
        super(Mydataset, self).__init__()
        self.path=path
        self.name=os.listdir(os.path.join(self.path,"segment"))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, item):
        name=self.name[item]
        namejpg=name[:-3]+"jpg"
        oriimg_path=os.path.join(self.path,"images")
        segimg_path=os.path.join(self.path,"segment")
        oriimg=cv2.imread(os.path.join(oriimg_path,namejpg))
        segimg=cv2.imread(os.path.join(segimg_path,name),0)

        h,w=oriimg.shape[0:2]
        size=max(h,w)
        ratio=512/size
        new_w=512 if w>h else w*ratio
        new_h=512 if h>w else h*ratio
        new_w,new_h=int(new_w),int(new_h)
        new_oriimg=cv2.resize(oriimg,(new_w,new_h),interpolation=cv2.INTER_LINEAR)
        new_segimg=cv2.resize(segimg,(new_w, new_h),interpolation=cv2.INTER_LINEAR)
        new_oriimg =cv2.copyMakeBorder(new_oriimg,0,512-new_h,0,512-new_w,cv2.BORDER_CONSTANT,value=0)
        new_segimg =cv2.copyMakeBorder(new_segimg,0,512-new_h,0,512-new_w,cv2.BORDER_CONSTANT,value=0)

        new_oriimg =cv2.cvtColor(new_oriimg,cv2.COLOR_BGR2RGB).transpose(2,0,1)/255-0.5
        new_segimg =new_segimg

        return torch.Tensor(new_oriimg),torch.Tensor(new_segimg)

if __name__ == '__main__':
    path="data"
    dataset=Mydataset(path)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i in data_loader:
        print(i[0].shape)
        print(i[1].shape)
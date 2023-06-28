from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import torch
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self,path,size):
        super(FaceDataset, self).__init__()
        self.size=size
        self.path=path
        self.dataset=[]
        self.dataset.extend(open(os.path.join(path,self.size,"positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path,self.size,"negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path,self.size,"part.txt")).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs=self.dataset[index].strip().split(" ")
        cond=torch.Tensor([int(strs[1])])
        offset=torch.Tensor([float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5])])
        landmark=torch.Tensor([float(strs[6]),float(strs[7]),float(strs[8]),float(strs[9]),float(strs[10]),float(strs[11]),float(strs[12]),float(strs[13]),float(strs[14]),float(strs[15])])

        img_path=os.path.join(self.path,self.size,strs[0])
        img_data=torch.Tensor(np.array(Image.open(img_path))/255-0.5)
        img_data=img_data.permute(2,0,1)

        return img_data,cond,offset,landmark

if __name__ == '__main__':
    path=r"D:\Relearn\Day006\python\MTCNN\data"
    dataset=FaceDataset(path,'48')
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i in data_loader:
        print(i[0].shape)
        print(i[1].shape)
        print(i[2].shape)
        print(i[3].shape)

import os
from torch.utils.data import Dataset,DataLoader
import cv2
import numpy as np

class MINISTDataset(Dataset):
    def __init__(self,root,is_train):
        super(MINISTDataset, self).__init__()
        self.dataset=[]
        sub_dir="TRAIN" if is_train else "TEST"
        for tag in os.listdir(f"{root}/{sub_dir}"):
            img_dir=f"{root}/{sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):
                img_path=f"{img_dir}/{img_filename}"
                self.dataset.append((img_path,tag))

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        data=self.dataset[index]
        img_data=cv2.imread(data[0],0)
        img_data=np.expand_dims(img_data,axis=0)
        img_data=img_data/255

        tag_one_hot=np.zeros(10)
        tag_one_hot[int(data[1])]=1

        return np.float32(img_data),np.float32(tag_one_hot)

if __name__ == '__main__':
    dataset=MINISTDataset("../diy_num_project/data",True)
    print(dataset[55555][1])
    data_loader=DataLoader(dataset,batch_size=10,shuffle=True)
    for i in data_loader:
        print(i[0].shape)
        print(i[1].shape)
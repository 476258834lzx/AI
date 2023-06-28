import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision
import os
from torchvision.utils import save_image
import numpy as np

transformer=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

class Mydataset(Dataset):
    def __init__(self,path):
        super(Mydataset, self).__init__()
        self.path=path
        self.name=os.listdir(os.path.join(self.path,"segment"))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, item):
        black0=torchvision.transforms.ToPILImage()(torch.zeros(3,512,512))
        black1=torchvision.transforms.ToPILImage()(torch.zeros(1,512,512))
        name=self.name[item]
        namejpg=name[:-3]+"jpg"
        oriimg_path=os.path.join(self.path,"images")
        segimg_path=os.path.join(self.path,"segment")
        oriimg=Image.open(os.path.join(oriimg_path,namejpg))
        segimg=Image.open(os.path.join(segimg_path,name))
        w,h=oriimg.size
        size=max(w,h)
        ratio=512/size
        new_w=512 if w>h else w*ratio
        new_h=512 if h>w else h*ratio
        new_w,new_h=int(new_w),int(new_h)
        new_oriimg=oriimg.resize((new_w,new_h))
        new_segimg=segimg.resize((new_w,new_h))
        black0.paste(new_oriimg,(0,0,new_w,new_h))
        black1.paste(new_segimg,(0,0,new_w,new_h))

        return transformer(black0),torch.tensor(np.array(black1,dtype=np.float32))

if __name__ == '__main__':
    path="data"
    dataset=Mydataset(path)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i in data_loader:
        print(i[0].shape)
        print(i[1].shape)
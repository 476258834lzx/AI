import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import os

classes=["cangjingkong","chenguanxi","chenweiting","dilireba","fanbingbing","huge","jialing","liuyifei","shayi","wujing","xietingfeng","yangmi","yueyunpeng","liudehua","pengshiliu"]

tf=transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,0.5,0.5),
        std=(0.5,0.5,0.5)
    )
])

class Face_dataset(Dataset):
    def __init__(self,root_dir="img",is_Train=True):
        super(Face_dataset, self).__init__()
        self.dataset=[]
        tag = "train" if is_Train else "val"
        for face_dir in os.listdir(root_dir):
            for face_filename in os.listdir(os.path.join(root_dir,face_dir,tag)):
                self.dataset.append([os.path.join(root_dir,face_dir,tag,face_filename),face_dir])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data=self.dataset[item]
        img_data=tf(Image.open(data[0]))
        label=torch.tensor(classes.index(data[1]))
        # label=F.one_hot(label,num_classes=len(classes)).float()
        return img_data,label

if __name__ == '__main__':
    face_dataset=Face_dataset("img",True)
    print(face_dataset[1000])
    print(face_dataset[1000][0].shape)
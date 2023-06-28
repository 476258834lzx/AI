import torch
from datasets import load_dataset
from torch.utils.data import Dataset

class My_Dataset(Dataset):
    def __init__(self,split):
        super(My_Dataset, self).__init__()
        #从磁盘加载CSV数据
        self.dataset=load_dataset(path="csv",data_files=f"data/{split}.csv",split="train")#官方bug
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text=self.dataset[index]["text"]
        label=self.dataset[index]["label"]
        return text,label

if __name__ == '__main__':
    dataset=My_Dataset("train")
    for data in dataset:
        print(data)
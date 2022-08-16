import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,split):
        super(MyDataset, self).__init__()
        self.dataset = load_from_disk("data/ChnSentiCorp")
        if split=="train":
            self.dataset=self.dataset['train']
        elif split=="validation":
            self.dataset=self.dataset['volidation']
        elif split=="test":
            self.dataset = self.dataset['test']
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        text=self.dataset[index]["text"]
        label=self.dataset[index]["label"]
        return text,label

if __name__ == '__main__':
    dataset=MyDataset(split="train")
    for data in dataset:
        print(data[1])
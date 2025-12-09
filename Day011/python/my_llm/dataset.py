from torch.utils.data import DataLoader, Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, file_path, batch_size):
        self.ids=torch.load(file_path)
        self.ids=self.ids[:self.ids.shape[0]//batch_size*batch_size].reshape(-1,batch_size)#每条一样长，多的不满一个batch的条数

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        return self.ids[idx]

if __name__ == '__main__':
    dataset = MyDataset('./data/2022-05_zh_middle_0010', batch_size=15)
    item=dataset[0]
    print(item)
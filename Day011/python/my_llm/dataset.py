from torch.utils.data import DataLoader, Dataset
import torch
import pickle

class PretrainDataset(Dataset):
    def __init__(self, file_path, paragraph_size):
        self.ids=torch.load(file_path)
        self.ids=self.ids[:self.ids.shape[0]//paragraph_size*paragraph_size].reshape(-1,paragraph_size)#每条一样长，多的不满一个batch的条数

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        return self.ids[idx]

class SftDataset(Dataset):
    def __init__(self, file_path, max_seq_len):
        self.max_seq_len = max_seq_len

        with open(file_path, "rb") as fr:
            self.datas = pickle.load(fr)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        prompt, tag = self.datas[index]

        prompt_len = len(prompt)

        if prompt_len <= self.max_seq_len:
            fill_zero = (self.max_seq_len-prompt_len)*[0,]
            prompt = prompt + fill_zero
            tag = tag + fill_zero
        else:
            prompt = prompt[:self.max_seq_len]
            tag = tag[:self.max_seq_len]

        return (torch.tensor(prompt, dtype=torch.long),
                torch.tensor(tag, dtype=torch.long))

if __name__ == '__main__':
    dataset = PretrainDataset('./data/2022-05_zh_middle_0010', paragraph_size=15)
    item=dataset[0]
    print(item)
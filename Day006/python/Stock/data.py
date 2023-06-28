from torch.utils.data import DataLoader,Dataset

class Stock(Dataset):
    def __init__(self):
        super(Stock, self).__init__()

    def __len__(self):
        return 0

    def __getitem__(self, item):
        return 0

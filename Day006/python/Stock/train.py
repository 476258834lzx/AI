import torch
from torch import nn
from net import Net
import os
from torch import optim
from data import Stock
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK']='True'
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
logs=SummaryWriter("logs")

class Trainer:
    def __init__(self):
        self._net=Net().to(DEVICE)
        self._opt=optim.Adam(self._net.parameters())
        self._train_dataset=Stock()
        self._train_dataloader=DataLoader(self._train_dataset,12,True)
        self._val_dataset = Stock()
        self._val_dataloader = DataLoader(self._val_dataset,4)
        self._loss_func=nn.CrossEntropyLoss()

    def __call__(self):
        for _epoch in range(1000000):
            self._net.train()
            _sum_loss=0
            for _i,(_data,_label) in enumerate(self._train_dataloader):
                _data,_label=_data.to(DEVICE),_label.to(DEVICE)
                _y=self._net(_data)
                _loss=self._loss_func(_y,_label)

                self._opt.zero_grad()#机器性能不够，隔X次step()一次，zero_grad()一次
                _loss.backward()
                self._opt.step()

                _sum_loss+=_loss.detach().cpu().ietm()

            logs.add_scalar("loss",_sum_loss/len(self._train_dataloader),_epoch)
            _sum_acc=0
            self._net.eval()
            for _i,(_data,_label) in enumerate(self._val_dataloader):
                _data = _data.to(DEVICE)
                _y = self._net(_data).detach().cpu().item()

                _sum_acc+=torch.sum(_label[_y>0.5]==1.)
            logs.add_scalar("acc",_sum_acc/len(self._val_dataloader),_epoch)





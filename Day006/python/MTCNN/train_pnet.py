from train import Trainer
from net import *


if __name__ == '__main__':
    trainer=Trainer(PNet,"12",r"params\pnet.pt",r"D:\Relearn\Day006\python\MTCNN\data",isCuda=True)
    trainer.train()
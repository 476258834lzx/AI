from train import Trainer
from net import *


if __name__ == '__main__':
    trainer=Trainer(RNet,"24",r"params\rnet.pt",r"D:\Relearn\Day006\python\MTCNN\data",isCuda=True)
    trainer.train()
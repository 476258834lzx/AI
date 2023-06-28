from train import Trainer
from net import *


if __name__ == '__main__':
    trainer=Trainer(ONet,"48",r"params\onet.pt",r"D:\Relearn\Day006\python\MTCNN\data",isCuda=True)
    trainer.train()
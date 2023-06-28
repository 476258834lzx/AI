import torch
from torch import jit
from net import NetV3


if __name__ == '__main__':
    model=NetV3()
    model.load_state_dict(torch.load("params/0.pth"))
    #虚拟输入
    input=torch.randn(1,784)

    traced_script_model=torch.jit.trace(model,input)
    traced_script_model.save("minist.pt")
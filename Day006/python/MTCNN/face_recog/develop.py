import torch
from torch import jit
from face_net import FaceNet

if __name__ == '__main__':
    model=FaceNet()
    model.load_state_dict(torch.load("params/best.pt"))
    #虚拟输入
    input=torch.randn(1,3,128,128)

    traced_script_model=torch.jit.trace(model,input)
    traced_script_model.save("face_recog.pt")
from utils.general import compare_cosion
from face_net import *
from torchvision import transforms
from PIL import Image
import torch

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf=transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,0.5,0.5),
        std=(0.5,0.5,0.5)
    )
])

net=FaceNet()
net.load_state_dict(torch.load("params/best.pt"))
net.to(DEVICE)
net.eval()

img1=tf(Image.open("test_img")).to(DEVICE)
vector1=net.register(img1[None,...])
img2=tf(Image.open("test_img")).to(DEVICE)
vector2=net.register(torch.unsqueeze(img2,dim=0))

simalarity=compare_cosion(vector1,vector2)
print(simalarity) 
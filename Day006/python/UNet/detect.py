#原图和mask绘画
import torch
from PIL import Image
import os
from net import net
import numpy as np
from torchvision import transforms

tf=transforms.Compose([
    transforms.ToTensor()
])

classes=["background","drivable_area","column"]

def label2image(pred, color):#N*3
    X = pred.long()#H*W
    return color[X, :]

class Detecter:
    def __init__(self, net_param,color,isCuda=True):
        self.isCuda = isCuda

        self.net = net()
        self.color=color

        self.net.load_state_dict(torch.load(net_param))

        if self.isCuda:
            self.net = self.net.cuda()

        self.net.eval()

    def detect(self, image):
        if self.isCuda:
            image = image.cuda()

        segimg=self.net(image)#N*3*512*512
        segimg=segimg.argmax(dim=1).cpu()#N*512*512
        mask=label2image(segimg,self.color)
        out=Image.blend(transforms.ToPILImage()(image.cpu().squeeze(0)),transforms.ToPILImage()(mask.squeeze(0)),0.4)#mask的alpha

        return out


if __name__ == '__main__':
    image_path = r"test_img"
    color = np.random.randint(0, 256, (len(classes), 3), dtype=np.uint8)
    detector = Detecter(r"weights\best.pt",color)
    for i in os.listdir(image_path):
        black = transforms.ToPILImage()(torch.zeros(3, 512, 512))
        img = Image.open(os.path.join(image_path, i))
        w, h = img.size
        size = max(w, h)
        ratio = 512 / size
        new_w = 512 if w > h else w * ratio
        new_h = 512 if h > w else h * ratio
        new_w, new_h = int(new_w), int(new_h)
        new_img = img.resize((new_w, new_h))
        black.paste(new_img, (0, 0, new_w, new_h))
        img_data=tf(black)
        seg=detector.detect(img_data.unsqueeze(dim=0))
        fusion=seg.resize((size,size),Image.BILINEAR)
        fusion=fusion.crop([0,0,w,h])
        fusion.show()
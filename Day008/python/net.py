import torch.nn as nn
import torch.nn.functional as F
import torch

class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.pre_layer=nn.Sequential(
            nn.Conv2d(3,10,(3,3),(1,1),(1,1)),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(10,16,(3,3),(1,1)),
            nn.PReLU(),
            nn.Conv2d(16,32,(3,3),(1,1)),
            nn.PReLU()
        )
        self.conv4_1=nn.Conv2d(32,1,(1,1),(1,1))
        self.conv4_2=nn.Conv2d(32,4,(1,1),(1,1))
        self.conv4_3=nn.Conv2d(32,10,(1,1),(1,1))
    def forward(self,x):
        feature=self.pre_layer(x)
        cond=torch.sigmoid(self.conv4_1(feature))
        bbox=self.conv4_2(feature)
        landmark=self.conv4_3(feature)
        return cond,bbox,landmark

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.pre_layer=nn.Sequential(
            nn.Conv2d(3,28,(3,3),(1,1),(1,1)),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(28,48,(3,3),(1,1)),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48,64,(2,2),(1,1)),
            nn.PReLU()
        )
        self.layer4=nn.Linear(64*3*3,128)
        self.pre_relu=nn.PReLU()

        self.layer5_1=nn.Linear(128,1)
        self.layer5_2=nn.Linear(128,4)
        self.layer5_3=nn.Linear(128,10)

    def forward(self,x):
        feature=self.pre_relu(self.layer4(self.pre_layer(x).reshape(x.size(0),-1)))
        cond=torch.sigmoid(self.layer5_1(feature))
        bbox=self.layer5_2(feature)
        landmark=self.layer5_3(feature)
        return cond,bbox,landmark

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.pre_layer=nn.Sequential(
            nn.Conv2d(3,32,(3,3),(1,1),(1,1)),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(32,64,(3,3),(1,1)),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64,64,(3,3),(1,1)),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (2, 2), (1, 1)),
            nn.PReLU()
        )
        self.layer5 = nn.Linear(128 * 3 * 3, 256)
        self.pre_relu = nn.PReLU()

        self.layer6_1 = nn.Linear(256, 1)
        self.layer6_2 = nn.Linear(256, 4)
        self.layer6_3 = nn.Linear(256, 10)
    def forward(self,x):
        feature = self.pre_relu(self.layer5(self.pre_layer(x).reshape(x.size(0), -1)))
        cond = torch.sigmoid(self.layer6_1(feature))
        bbox = self.layer6_2(feature)
        landmark = self.layer6_3(feature)
        return cond,bbox,landmark

if __name__ == '__main__':
    img=torch.randn(2,3,48,48)
    net=ONet()
    print(net(img)[0].shape)
    print(net(img)[1].shape)
    print(net(img)[2].shape)

from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(3,64,(3,3),(2,2),(1,1),bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128,256 , (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        self.fc=nn.Linear(32*32*256,1024)

    def forward(self,x):
        featuremap=self.layer(x)
        featuremap=featuremap.reshape(-1,32*32*256)
        vector=self.fc(featuremap)
        return vector

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(1024, 32 * 32 * 256),
            # nn.BatchNorm1d(32 * 32 * 256),
            # nn.PReLU()
        )
        self.layer=nn.Sequential(
            nn.ConvTranspose2d(256,128,(3,3),(1,1),(1,1),bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, (3, 3), (2, 2), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 3, (3, 3), (2, 2), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(3),
            nn.PReLU()
        )

    def forward(self,x):
        fc=self.fc(x)
        fc=fc.reshape(-1,256,32,32)
        out=self.layer(fc)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()

    def forward(self,x):
        vector=self.encoder(x)
        out=self.decoder(vector)

        return out

if __name__ == '__main__':
    net=Net()
    x=torch.randn(1,3,128,128)
    out=net(x)
    print(out.shape)
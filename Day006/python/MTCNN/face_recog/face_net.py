from torchvision import models
from loss import *
class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.sub_net=models.densenet121(pretrained=True)
        self.sub_net.classifier=nn.Linear(1024,512,bias=False)
        self.arg_softmax=ArcsoftmaxLoss(512,15)

    def forward(self,x):
        feature=self.sub_net(x)
        out=self.arg_softmax(feature,1,1)
        return feature,out

    def register(self,x):
        return self.sub_net(x)

if __name__ == '__main__':
    x=torch.randn(1,3,256,256)
    net=FaceNet()
    print(net(x)[0].shape)
    print(net(x)[1].shape)
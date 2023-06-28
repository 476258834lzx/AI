from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from net import NetV1
from torch import optim,nn
from torch.utils.tensorboard import SummaryWriter

train_dataset=datasets.MNIST(root="MNIST_data",train=True,transform=transforms.ToTensor,download=True)
test_dataset=datasets.MNIST(root="MNIST_data",train=False,download=True)
train_dataloader=DataLoader(train_dataset,1,True)
test_dataloader=DataLoader(test_dataset,1,True)

if __name__ == '__main__':
    Summary=SummaryWriter("logs")
    net = NetV1()
    opt = optim.Adam(net.parameters())
    loss_func = nn.MSELoss()
    for epoch in range(100):
        for i,(img,tag)in enumerate(train_dataloader):
            net.train()
            y=net(img.reshape(-1,784))
            tag=one_hot(tag,10).float()

            loss=loss_func(y,tag)
            Summary.add_scalar("train_loss",loss,epoch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(loss.item())
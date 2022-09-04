from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

train_dataset=datasets.MNIST(root="MNIST_data",train=True,transform=transforms.ToTensor,download=True)
test_dataset=datasets.MNIST(root="MNIST_data",train=False,download=True)

# img_data=train_dataset.data[0]
# unloader=transforms.ToPILImage()
# img=unloader(img_data)
# print(type(img))
# img.show()
train_dataloader=DataLoader(train_dataset,1,True)
test_dataloader=DataLoader(test_dataset,1,True)

for i,(img,tag)in enumerate(train_dataloader):
    print(img)
    print(one_hot(tag,10))
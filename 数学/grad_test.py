import torch

x=torch.tensor([3.0],requires_grad=True)
y=x**2+2
# y.backward()
# print(x.grad)

print(torch.autograd.grad(y,x))

import torch
from torch.nn import functional as F

embs=torch.randn(5,2)
tokens=torch.tensor([1,2,1,0,0])#就是稠密编码的索引

a=F.one_hot(tokens,5).float()

print(embs)
print(a)
print(a@embs)
print(embs[tokens])
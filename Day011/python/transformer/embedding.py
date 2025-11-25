import torch
from torch.nn import functional as F

embs=torch.nn.Embedding(5,2)
tokens=torch.tensor([1,2,1,0,0])
print(embs.weight.data)
print(embs(tokens))
import torch
import numpy as np

a=np.array([[1,2],[3,4]])
print(a.ndim)
print(a.shape)

b=torch.tensor([[1,2],[3,4]])
print(b.size())
print(b.shape)
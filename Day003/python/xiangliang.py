import torch
import numpy as np

a=np.array([1,2])
print(a.ndim)

b=torch.tensor([1,2])
print(b.size())
print(b.shape)
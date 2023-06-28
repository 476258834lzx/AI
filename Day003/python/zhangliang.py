import torch
import numpy as np

a=np.array([[[1,2,3],[4,5,6],[7,8,9]],[[9,8,7],[6,5,4],[3,2,1]],[[1,2,3],[4,5,6],[7,8,9]]])
print(a.ndim)
print(a.shape)

b=torch.tensor([[[1,2,3],[4,5,6],[7,8,9]],[[9,8,7],[6,5,4],[3,2,1]],[[1,2,3],[4,5,6],[7,8,9]]])
print(b.size())
print(b.shape)
print(b.T)
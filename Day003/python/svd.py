import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
a=torch.tensor([[1.,2.],[3.,4.]])
b=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
print(torch.eig(a,eigenvectors=True))#相似矩阵
print(torch.svd(b))

c=np.array([[1.,2.],[3.,4.],[5.,6.]])
print(np.linalg.svd(c))
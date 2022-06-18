import torch
import numpy as np

a=np.random.randint(low=0,high=10,size=[2,3,2])
b=np.random.randint(0,10,[2,2,3])
print(a)
print(b)
print(a*a)
print(a*3)
#numpy不支持张量外积

c=torch.randint(10,[2,3,2])
d=torch.randint(10,[2,2,3])
print(torch.matmul(c,d))
print(c@d)

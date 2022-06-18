import numpy as np

x=np.matrix(np.array([[3],[1],[6]]))
y=4*x

print((x.T@x).I@x.T@y)
print(np.linalg.inv(x.T@x)@x.T@y)
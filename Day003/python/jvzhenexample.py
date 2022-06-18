import numpy as np#torch相同

#单位阵
a=np.eye(4)
print(a)
a=np.eye(3,4)
print(a)

#对角阵
a=np.diag([1,2,3,4])
print(a)

#三角矩阵
a=np.tri(3,3)
print(a)
a=np.triu(np.array([[1,2,3],[3,4,5],[4,5,6]]))
print(a)
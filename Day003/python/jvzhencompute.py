import numpy as np

a=np.array([[1,2],[3,4],[5,6]])
b=np.array([[1,2],[3,4],[5,6]])

print(a+b)
print(a+3)
print(a*b)
print(a*3)

c=np.array([[1,2,3],[3,4,5]])
print(np.dot(a,c))
print(a.dot(c))
print(a@c)

print(a.T)  
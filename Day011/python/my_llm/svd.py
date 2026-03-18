# import numpy as np
#
# # 创建一个矩阵 A
# A = np.array([[1, 2, 3], [4, 5, 6],[7,8,9]])
#
# # 执行奇异值分解
# U, s, V = np.linalg.svd(A, full_matrices=True)
#
# # s[-1]=0
# # print("左奇异矩阵 U:\n", U)
# # print("奇异值 s:\n", s)
# # print("右奇异矩阵 V^T:\n", V)
#
# s[-1]=0
# s[-2]=0
# B = U@np.diag(s)@V
# print(B)
#
#
# x = np.array([[1,2,3]])
# # print(x.shape)
#
# y1 = x@A
# y2 = x@B
#
# print(y1)
# print(y2)
#
# # _A = U@np.diag(s)@V
# # # print(_A)
# # x = np.random.randn(3,3)
# # print(x@A)
# # print(x@_A)
# # loss = (x@A - x@_A).sum()
# # print(loss)


from matplotlib import pyplot as plt
from numpy import linalg as LA
import numpy as np

mat = plt.imread("1.jpg")
mat = np.dot(mat[...,:3], [0.2989, 0.5870, 0.1140])

# SVD
U, s, VT = LA.svd(mat)

Sigma = np.zeros((mat.shape[0], mat.shape[1]))
Sigma[:min(mat.shape[0], mat.shape[1]), :min(mat.shape[0], mat.shape[1])] = np.diag(s)

# Reconstruction of the matrix using the first 30 singular values

diffs = np.diff(s)
k_adaptive = np.argmax(diffs) + 1  # 找下降最慢的点

mat_approx = U[:, :k_adaptive] @ Sigma[:k_adaptive, :k_adaptive] @ VT[:k_adaptive, :]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,8))
plt.subplots_adjust(wspace=0.3, hspace=0.2)

ax1.imshow(mat, cmap='gray')
ax1.set_title("Original image")

ax2.imshow(mat_approx, cmap='gray')
ax2.set_title("Reconstructed image using the \n first {} singular values".format(k_adaptive))
plt.show()
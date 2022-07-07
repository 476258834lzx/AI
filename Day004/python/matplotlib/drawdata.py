from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

# img=Image.open("../img/5.jpg")
# plt.imshow(img)
# plt.axis(False)
# plt.show()

# x=np.random.randn(20)
# y=np.random.randn(20)
#绘制点图
# plt.plot(x,y,".")
# plt.scatter(x,y,label="like",c="blue",marker="*")
#显示图例
# plt.legend()
# plt.show()
#################动态画图plt.ion、plt.cla、plt.ioff

#绘制3D图
from mpl_toolkits.mplot3d import Axes3D
x=np.random.randn(100)
y=np.random.randn(100)
z=np.random.randn(100)

fig=plt.figure()
ax=Axes3D(fig)
# ax.plot(x,y,z,"*")
ax.plot3D(x,y,z,".")
# ax.scatter(x,y,z)
plt.show()

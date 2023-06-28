import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread("../img/12.jpg",0)
laplacian=cv2.Laplacian(img,cv2.CV_64F)
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)#原图，深度，x方向布尔值，y方向布尔值，卷积核尺寸
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)#原图，深度，x方向布尔值，y方向布尔值，卷积核尺寸
plt.subplot(2,2,1)#图片形状，图像位置索引
plt.imshow(img,cmap="gray")##########?
plt.title("Original")
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)#图片形状，图像位置索引
plt.imshow(laplacian,cmap="gray")
plt.title("laplacian")
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)#图片形状，图像位置索引
plt.imshow(sobelx,cmap="gray")
plt.title("sobelx")
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)#图片形状，图像位置索引
plt.imshow(sobely,cmap="gray")
plt.title("sobely")
plt.xticks([])
plt.yticks([])
plt.show()
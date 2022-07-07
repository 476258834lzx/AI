import numpy as np
import cv2

# x=np.uint8([250])
# y=np.uint8([10])
# print(cv2.add(x,y))#像素取值不会超过0-255
# print(cv2.subtract(y,x))#像素取值不会超过0-255

img1=cv2.imread("../img/10.jpg")
img2=cv2.imread("../img/11.jpg")
dst=cv2.addWeighted(img1,0.9,img2,0.1,0)#最后0像素修正值
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

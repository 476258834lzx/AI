import cv2
import numpy as np
#取出HSV色度、饱和度、亮度的特定范围的图像
img=cv2.imread("../img/3.jpg")
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower_blue=np.array([100,200,100])
upper_blue=np.array([200,255,200])

mask=cv2.inRange(hsv,lower_blue,upper_blue)
res=cv2.bitwise_and(img,img,mask=mask)

cv2.imshow('frame',img)
cv2.imshow('mask',res)

cv2.waitKey(0)
cv2.destroyAllWindows()
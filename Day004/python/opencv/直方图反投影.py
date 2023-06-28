import cv2
import numpy as np

roi=cv2.imread("../img/13.jpg")
hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
target=cv2.imread("../img/1.jpg")
hsvt=cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

roihist=cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)#对原图做归一化
#直方图反向投影
dst=cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)#原图，通道，直方图，取值范围，大小
disc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))#获取滤波核
#在小区域反投影的结果上进行滤波
dst=cv2.filter2D(dst,-1,disc)#滤波图像，所有通道，滤波核
ret,thresh=cv2.threshold(dst,50,255,0)
thresh=cv2.merge((thresh,thresh,thresh))
res=cv2.bitwise_and(target,thresh)
res=np.hstack((target,thresh,res))#纵向拼接
cv2.imwrite("hist.jpg",res)
cv2.imshow("hist",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
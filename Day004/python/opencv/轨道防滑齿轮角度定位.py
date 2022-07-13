import cv2
import numpy as np
img=cv2.imread("../img/15.jpg",0)

kernel=np.array([[0,1,0],[-1,5,-1],[0,-1,0]],np.float32)#拉普拉斯核
img=cv2.filter2D(img,-1,kernel=kernel)
img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)#操作图，像素上限，求阈值的方法，二值图类型，相邻区大小，加权修正系数
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))#矩形核
img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel,iterations=1)
cv2.imshow("dst",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
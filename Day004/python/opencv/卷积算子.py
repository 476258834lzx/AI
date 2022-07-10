import cv2
import numpy as np

src=cv2.imread("../img/10.jpg")
#滤波器(卷积核)
kernel=np.array([[1,1,0],[1,0,-1],[0,-1,-1]],dtype=np.float32)
kernel1=np.array([[4,0,0],[0,0,0],[0,0,-4]],dtype=np.float32)
kernel2=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)#锐化算子
#滤波操作/卷积操作
dst=cv2.filter2D(src,-1,kernel)#原图，卷积通道(-1指卷所有通道)，卷积核

cv2.imshow("img",src)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
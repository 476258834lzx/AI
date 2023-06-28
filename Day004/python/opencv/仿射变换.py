import numpy as np

import cv2

src=cv2.imread("../img/4.jpg")
rows,cols,channel=src.shape
# dst=cv2.resize(src,(cols*2,rows*2),interpolation=cv2.INTER_CUBIC)
# dst=cv2.transpose(src)
# dst=cv2.flip(src,1)

#定义仿射矩阵
# M=np.float32([[1,0,50],[0,1,50]])#平移,比自己写的sheer方便多了
# M=np.float32([[0.5,0,0],[0,0.5,0]])#缩放
# M=np.float32([[-0.5,0,cols//2],[0,0.5,0]])#对称平移
M=cv2.getRotationMatrix2D((cols//2,rows//2),45,0.7)#获取缩放旋转仿射矩阵

dst=cv2.warpAffine(src,M,(cols,rows))
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
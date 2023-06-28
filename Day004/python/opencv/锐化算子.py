import cv2
import numpy as np

src=cv2.imread(r"../img/1.jpg")
#拉普拉斯锐化算子
# kernel=np.array([[0,1,0],[-1,5,-1],[0,-1,0]],np.float32)#拉普拉斯核
# dst=cv2.filter2D(src,-1,kernel=kernel)#原图，深度，卷积核
#USM锐化
# dst=cv2.GaussianBlur(src,(5,5),0)
# dst=cv2.addWeighted(src,2,dst,-1,0)

#拉普拉斯滤波算子
dst=cv2.Laplacian(src,-1,(3,3))
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
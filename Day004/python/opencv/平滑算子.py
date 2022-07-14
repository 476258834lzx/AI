import cv2

src=cv2.imread("../img/2.jpg")
#均值滤波
# dst=cv2.blur(src,(5,5))#卷积核大小
#高斯滤波
# dst=cv2.GaussianBlur(src,(5,5),sigmaX=0)#卷积核大小，X轴方向模糊度标准差
#中值滤波
# dst=cv2.medianBlur(src,5)#滤波算子形状尺寸
#双边滤波
dst=cv2.bilateralFilter(src,9,75,75)#断裂点间的距离，空间高斯函数标准差，灰度值相似度高斯函数标准差（取值范围0-255，通常取值75-125之间）

cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
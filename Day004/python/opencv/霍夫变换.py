import cv2
import numpy as np

img=cv2.imread("../img/16.jpg",0)
#提取轮廓或者边缘
edges=cv2.Canny(img,50,150)
#霍夫直线检测,返回直线列表
lines=cv2.HoughLinesP(edges,1.0,np.pi/180,400)#老版本(二值图或灰度图),新版必须二值图,rho参数线段以像素为单位的精度,theta弧度精度,霍夫空间中过该点的直线的个数,最小直线长度，点与点的最大长度
lines=np.squeeze(lines)
for x1,y1,x2,y2 in lines:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
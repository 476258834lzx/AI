import cv2
import numpy as np

img=cv2.imread("../img/12.jpg")
#透视前的坐标
pts1=np.float32([[53,50],[369,39],[27,296],[390,301]])
pts2=np.float32([[0,0],[413,0],[0,323],[413,323]])
M=cv2.getPerspectiveTransform(pts1,pts2)
dst=cv2.warpPerspective(img,M,(413,323))
cv2.imshow("img",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
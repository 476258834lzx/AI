import cv2
from matplotlib import pyplot as plt

img=cv2.imread("../img/10.jpg",0)
# img_B=cv2.calcHist([img],[0],None,[256],[0,256])#原图，通道，掩码，取值范围，列出范围
# plt.plot(img_B,label='B',color='b')
# img_G=cv2.calcHist([img],[1],None,[256],[0,256])
# plt.plot(img_G,label='B',color='g')
# img_R=cv2.calcHist([img],[2],None,[256],[0,256])
# plt.plot(img_R,label='B',color='r')
#
# plt.show()

# dst=cv2.equalizeHist(img)
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))#自适应均衡化的比例
dst=clahe.apply(img)
# dst=cv2.subtract(dst,img)
cv2.imshow("img",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2

img=cv2.imread("../img/10.jpg",0)
ret,img=cv2.threshold(img,80,255,cv2.THRESH_BINARY)
#定义核的形状
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))#矩形核
#膨胀
# dst=cv2.dilate(img,kernel)
#腐蚀
# dst=cv2.erode(img,kernel)
#开操作
# dst=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel,iterations=1)#迭代次数，进行开操作的次数
#闭操作
# dst=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,iterations=1)#迭代次数，进行闭操作的次数
#梯度操作
# dst=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel,iterations=1)#迭代次数，进行闭操作的次数
#黑帽操作
# dst=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel,iterations=1)#迭代次数，进行闭操作的次数
#礼帽操作
dst=cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel,iterations=1)#迭代次数，进行闭操作的次数
cv2.imshow("img",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
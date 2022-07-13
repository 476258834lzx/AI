import cv2

img=cv2.imread("../img/11.jpg")
canny=cv2.Canny(img,50,150)#原图，双边滤波器的两个阈值
cv2.imshow('img',img)
cv2.imshow('Canny',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
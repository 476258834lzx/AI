import cv2

img=cv2.imread("../img/14.jpg")
#对原图做高亮处理
img=cv2.convertScaleAbs(img,alpha=6,beta=0)
#高斯模糊
img=cv2.GaussianBlur(img,(5,5),0)

ret,img=cv2.threshold(img,80,255,cv2.THRESH_BINARY)
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,iterations=5)

img=cv2.Canny(img,50,150)
cv2.imshow("src",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
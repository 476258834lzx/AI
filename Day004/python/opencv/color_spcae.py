import cv2

src="../img/2.jpg"

img=cv2.imread(src)
# dst =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图
dst =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img[:,:,2]=0
cv2.imshow("img show",img)
cv2.imshow("dst show",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2

src=r"img/1.jpg"

img=cv2.imread(src)
cv2.imshow("pic",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
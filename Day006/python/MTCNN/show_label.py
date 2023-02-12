import cv2

image=r"F:\data\CelebA\Img\img_celeba.7z\img_celeba\000001.jpg"

img=cv2.imread(image)
cv2.circle(img,(165,184),5,(0,0,255),1)#中心点，半径，颜色，线宽
cv2.circle(img,(244,176),5,(0,0,255),1)#中心点，半径，颜色，线宽
cv2.circle(img,(196  ,249),5,(0,0,255),1)#中心点，半径，颜色，线宽
cv2.circle(img,(194  ,271),5,(0,0,255),1)#中心点，半径，颜色，线宽
cv2.circle(img,(266  ,260),5,(0,0,255),1)#中心点，半径，颜色，线宽

cv2.imshow("asd",img)
cv2.waitKey(0)
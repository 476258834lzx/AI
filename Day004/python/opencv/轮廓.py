import cv2

img=cv2.imread("../img/10.jpg")
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(imggray,0,255,0|cv2.THRESH_OTSU)

contours,image=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print(contours)

img_contours=cv2.drawContours(img,contours,-1,(0,255,0),1)#原图，轮廓，轮廓索引(-1绘制所有轮廓)，颜色，线宽(-1时填充)
cv2.imshow("img",img_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

#求中心
M=cv2.moments(contours[0])#第一组轮廓的矩
cx,cy=int(M['m10']/M['m00']),int(M['m01']/M['m00'])
print("重心:",cx,cy)
area=cv2.contourArea(contours[0])
print("面积:",area)

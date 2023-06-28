import cv2

img=cv2.imread("../img/10.jpg")
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(imggray,0,255,0|cv2.THRESH_OTSU)

contours,image=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print(contours)

#轮廓近似
# epsilon=20#近似度
# approx=cv2.approxPolyDP(contours[0],epsilon,True)
# img_contours=cv2.drawContours(img,[approx],-1,(0,255,0),1)#原图，轮廓，轮廓索引(-1绘制所有轮廓)，颜色，线宽
#检测并修复凸性缺陷
# hull=cv2.convexHull(contours[0])
# print(cv2.isContourConvex(contours[0]),cv2.isContourConvex(hull))
# img_contours=cv2.drawContours(img,[hull],-1,(0,255,0),1)#原图，轮廓，轮廓索引(-1绘制所有轮廓)，颜色，线宽
#边界矩形
# x,y,w,h=cv2.boundingRect(contours[0])
#最小外接矩形
# rect=cv2.minAreaRect(contours[0])
# box=cv2.boxPoints(rect)
# box=int(box)
#最小外接圆
(x,y),radius=cv2.minEnclosingCircle(contours[0])
center=(int(x),int(y))
radius=int(radius)
img_contours=cv2.circle(img,center,radius,(255,0,0),1)
cv2.imshow("img",img_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
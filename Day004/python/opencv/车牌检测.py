import cv2

rawImg=cv2.imread("../img/17.jpg")
#高斯模糊
image=cv2.GaussianBlur(rawImg,(3,3),0)
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
sobelx=cv2.Sobel(image,cv2.CV_16S,1,0)
#转回int8
image=cv2.convertScaleAbs(sobelx)
#二值化
ret,img=cv2.threshold(image,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
#补洞
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(17,5))
img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,iterations=2)
#形态学操作去噪
kernelx=cv2.getStructuringElement(cv2.MORPH_RECT,(20,1))
kernely=cv2.getStructuringElement(cv2.MORPH_RECT,(1,20))
img=cv2.dilate(img,kernelx,iterations=1)
img=cv2.erode(img,kernelx,iterations=1)
img=cv2.erode(img,kernely,iterations=1)
img=cv2.dilate(img,kernely,iterations=1)
#平滑处理
img=cv2.medianBlur(img,25)
#查找轮廓
contours,_=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for item in contours:
    rect=cv2.boundingRect(item)
    x=rect[0]
    y=rect[1]
    w=rect[2]
    h=rect[3]
    if w>(4*h):
        #裁剪区域
        dst=rawImg[y:y+h,x:x+w,:]
        cv2.imshow("dst"+str(x), dst)
#绘制轮廓
dst1=cv2.drawContours(rawImg,contours,-1,(0,0,255),1)
cv2.imshow("dst1",dst1)
cv2.waitKey(0)
cv2.destroyAllWindows()
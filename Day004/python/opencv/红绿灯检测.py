import cv2
import numpy as np

def tl_detection(rawImg):
    #高斯模糊
    image=cv2.GaussianBlur(rawImg,(5,5),3)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sobelx=cv2.Sobel(image,cv2.CV_64F,1,0)
    #转回int8
    image=cv2.convertScaleAbs(sobelx)
    #二值化
    ret,img=cv2.threshold(image,127,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #补洞
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(17,5))
    img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,iterations=2)
    #形态学操作去噪
    kernelx=cv2.getStructuringElement(cv2.MORPH_RECT,(22,1))
    kernely=cv2.getStructuringElement(cv2.MORPH_RECT,(1,35))
    img=cv2.dilate(img,kernelx,iterations=1)
    img=cv2.erode(img,kernelx,iterations=1)
    img=cv2.erode(img,kernely,iterations=1)
    img=cv2.dilate(img,kernely,iterations=1)
    #平滑处理
    img=cv2.medianBlur(img,17)
    #查找轮廓
    contours,_=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #颜色空间
    lower_hsv_red=np.array([0,255,135])
    upper_hsv_red=np.array([0,149,255])
    lower_hsv_yellow = np.array([18,255,130])
    upper_hsv_yellow = np.array([26,208,255])
    lower_hsv_green = np.array([71,206,255])
    upper_hsv_green = np.array([52,226,244])

    #侦测框列表
    rectlist=[]
    #提取边界矩形
    for item in contours:
        rect=cv2.boundingRect(item)
        x,y,width,height=rect[0],rect[1],rect[2],rect[3]
        if height<(width*2.5) and height>(width*2) and width>10:
            a=rawImg[y:y+height,x:x+width,:]
            cv2.imshow("dst" + str(x), a)
            a=cv2.cvtColor(a,cv2.COLOR_BGR2HSV)
            mask_red=cv2.inRange(a,lowerb=lower_hsv_red,upperb=upper_hsv_red)
            mask_yellow=cv2.inRange(a,lowerb=lower_hsv_yellow,upperb=upper_hsv_yellow)
            mask_green=cv2.inRange(a,lowerb=lower_hsv_green,upperb=upper_hsv_green)

            #筛选框颜色
            if np.max(mask_red)==255:
                rectlist.append((x,y,x+width,y+height,(0,0,255)))
            elif np.max(mask_yellow)==255:
                rectlist.append((x,y,x+width,y+height,(0,255,255)))
            elif np.max(mask_green)==255:
                rectlist.append((x,y,x+width,y+height,(0,255,0)))
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

    return rectlist,contours

if __name__ == '__main__':
    rawImg=cv2.imread("../img/18.jpg")
    rectlist,contours=tl_detection(rawImg)
    print(rectlist)
    for i in rectlist:
        cv2.rectangle(rawImg,(i[0],i[1]),(i[2],i[3]),i[4],thickness=1)
    cv2.imshow("image",rawImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
import cv2
import numpy as np

#HSV提取皮肤
def skin_detection_HSV(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #皮肤色调阈值
    lower_blue=np.array([0,18,102])
    upper_blue=np.array([17,133,242])
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    img=cv2.bitwise_and(frame,frame,mask=mask)
    return img

#直方图反投影提取皮肤
def skin_demo(roi):#获取样本直方图
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    roihist=cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    return roihist
def skin_detection_pro(roihist,frame):
    hsvt=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    dst=cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

    #补空洞
    disc=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dst=cv2.filter2D(dst,-1,disc)

    #二值化处理
    _,thresh=cv2.threshold(dst,150,255,0)

    #合并通道变为3通道
    thresh=cv2.merge((thresh,thresh,thresh))
    res=cv2.bitwise_and(frame,thresh)
    return res

if __name__ == '__main__':
    img=cv2.imread("../img/19.jpg")
    # res=skin_detection_HSV(img)
    res=skin_detection_pro(skin_demo(img),img)
    cv2.imshow("res",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
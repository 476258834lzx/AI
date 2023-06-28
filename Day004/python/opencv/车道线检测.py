import cv2
import numpy as np

def lane_detection(img,pt):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #创建蒙版
    mask=np.zeros(gray.shape,dtype=np.uint8)
    #填充指定区域形成掩码
    cv2.fillConvexPoly(mask,pt,1)
    gray_area=cv2.bitwise_and(gray,gray,mask=mask)

    _,thresh=cv2.threshold(gray_area,130,145,0)
    #霍夫直线检测
    lines=cv2.HoughLinesP(thresh,1,np.pi/180,30,maxLineGap=50)
    #无直线跳过
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line[0]
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
    return thresh,img

if __name__ == '__main__':
    img=cv2.imread("../img/16.jpg")
    pt=np.array([[649,411],[991,394],[1272,548],[1269,845],[882,842]])
    thresh,frame=lane_detection(img,pt)
    cv2.imshow("dst",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
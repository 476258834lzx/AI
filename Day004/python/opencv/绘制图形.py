import numpy as np
import cv2

img=cv2.imread(r"../img/9.jpg")

# cv2.line(img,(100,30),(210,180),(0,0,255),2)
# cv2.circle(img,(50,50),30,(0,0,255),1)#中心点，半径，颜色，线宽
# cv2.rectangle(img,(100,30),(210,180),color=(0,0,255),thickness=2)#左上角，右下角，颜色，线宽-1填充
# cv2.ellipse(img,(100,100),(100,50),45,270,360,(0,0,255),-1)#中心点，横向半径*纵向半径，倾斜角度，起始角度，终止角度，颜色，线宽
#
# pts=np.array([[10,5],[50,10],[70,20],[20,30],[60,30],[20,40]],np.int32)
# cv2.polylines(img,[pts],True,(0,0,255),2)#多点列表，是否闭合，颜色，线宽
#
#加载字体
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,"fucking bitch!",(100,30),font,1,(0,0,255),1,lineType=cv2.LINE_AA)#字符，字符左上角，字体，字体间距，颜色，线宽，像素补值法防锯齿
#
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
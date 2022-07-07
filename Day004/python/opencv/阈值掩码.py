import cv2

img=cv2.imread(r"../img/10.jpg",0)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#读入时取模式选0直接读取灰度图
# ret,binary=cv2.threshold(img,80,255,cv2.THRESH_BINARY)#操作图,阈值，像素上限,返回值ret为取的阈值
# ret,binary=cv2.threshold(img,80,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)#操作图,阈值无效，像素上限,返回值ret为取的阈值
# print(ret)

# binary=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,0)#操作图，像素上限，求阈值的方法，二值图类型，相邻区大小，加权修正系数
binary=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)#操作图，像素上限，求阈值的方法，二值图类型，相邻区大小，加权修正系数
cv2.imshow("img",binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
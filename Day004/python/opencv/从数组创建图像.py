import cv2
import numpy as np

img=np.zeros([200,200,3],dtype=np.uint8)
img1=np.zeros((200,200,3),dtype=np.uint8)

img[...,2]=255
img1[:,:,2]=255
print(img)
print(img1)
cv2.imwrite("save.jpg",img)#含有中文写入不了

cv2.imshow("src",img)#BGR
cv2.waitKey(0)
cv2.destroyAllWindows()


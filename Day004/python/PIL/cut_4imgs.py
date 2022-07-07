from PIL import Image
import numpy as np

img=Image.open("../img/6.jpg")
w,h=img.size
img=img.resize((w//2*2,h//2*2))
img_data=np.array(img)
img_data=img_data.reshape((2,h//2,2,w//2,3))
img_data=img_data.transpose(0,2,1,3,4)
img_data=img_data.reshape(-1,h//2,w//2,3)
img_data_1=img_data[0]
img_data_2=img_data[1]
img_data_3=img_data[2]
img_data_4=img_data[3]

img1=Image.fromarray(img_data_1,mode="RGB")
img2=Image.fromarray(img_data_2,mode="RGB")
img3=Image.fromarray(img_data_3,mode="RGB")
img4=Image.fromarray(img_data_4,mode="RGB")

img1.show()
img2.show()
img3.show()
img4.show()
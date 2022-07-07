import numpy as np
from PIL import Image

img1=Image.open("../img/2.jpg")
#图像模式
w,h=img1.size
print(w,h)
print(img1.getbands())
print(img1.mode)

#获取像素
pixel=img1.getpixel((100,100))
print(pixel)
# img=img1.convert("L")
# img.show()

#缩放
# img=img1.resize((300,1200))
# img.show()

#保存图像
# img.save("save.jpg")

#抠图
# img=img1.crop((150,150,260,260))
# img.show()

#旋转
# img=img1.rotate(45)
# img.show()

#粘贴
# img1.paste(Image.open("../img/10.jpg"))
# img1.show()

#翻转
# img=img1.transpose(Image.FLIP_LEFT_RIGHT)
# img.show()

#画图
# from PIL import ImageDraw
# draw=ImageDraw.Draw(img1)
# draw.rectangle((100,100,200,200),outline="red",width=1)
# draw.ellipse((100,100,200,200),outline="red")
# img1.show()

#滤波器
# from PIL import ImageFilter
# # img=img1.filter(ImageFilter.CONTOUR)
# # img=img1.filter(ImageFilter.BLUR)
# # img=img1.filter(ImageFilter.GaussianBlur)
# # img=img1.filter(ImageFilter.DETAIL)
# img=img1.filter(ImageFilter.EMBOSS)
# img.show()

#图像转矩阵
import numpy as np
img_data=np.array(img1)
print(img_data.shape)#HWC
print(img_data[:,:,0].shape)
#####拼接第三维度的方式
# B_data=np.expand_dims(img_data[:,:,0],axis=2)
# print(B_data.shape)
# mask=np.zeros((h,w,1))
# ThirdD_B_data=np.concatenate((mask,mask,B_data),axis=2)
# print(type(ThirdD_B_data))
# ThirdD_B_data=np.array(ThirdD_B_data,dtype=np.uint8)
#######构造三维图填充的方式
ThirdD_B_data=np.zeros((h,w,3),dtype=np.uint8)
mask=np.zeros((h,w),dtype=np.uint8)
B_data=img_data[:,:,0]
ThirdD_B_data[:,:,0]=mask
ThirdD_B_data[:,:,1]=mask
ThirdD_B_data[:,:,2]=B_data
new_img=Image.fromarray(ThirdD_B_data)
print(new_img.mode)
new_img.show()
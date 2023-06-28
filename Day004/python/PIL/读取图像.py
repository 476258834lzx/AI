from PIL import Image

src=r"../img/1.jpg"

img=Image.open(src)
print(type(img))
img.show()#使用当前系统默认图像查看器

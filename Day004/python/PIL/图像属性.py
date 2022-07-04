from PIL import Image

img1=Image.open("../img/2.jpg")
w,h=img1.size
print(w,h)
print(img1.getbands())
print(img1.mode)
pixel=img1.getpixel((100,100))
print(pixel)
img=img1.convert("L")
img.show()
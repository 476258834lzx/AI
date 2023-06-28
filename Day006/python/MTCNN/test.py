import os
import glob
import shutil
import numpy as np
import xml.etree.cElementTree as et

file_src=r"F:\data\CelebA\Img\img_celeba.7z\img_celeba\outputs"
anno_path=r"F:\data\CelebA\Img\img_celeba.7z\list_bbox_celeba.txt"
anno=open(anno_path,"a")
anno.write("202599\nimage_id x_1 y_1 width height\n")

for file_path in os.listdir(file_src):
    # print(file_path)
    img_name=file_path[:-4]+".jpg"
    tree=et.parse(f"{file_src}/{file_path}")
    root = tree.getroot()
    outputs=root.find("outputs")
    Object=outputs.find("object")

    for obj in Object.iter("item"):
        xmin=np.float32(obj.find("bndbox").find("xmin").text)
        ymin=np.float32(obj.find("bndbox").find("ymin").text)
        xmax=np.float32(obj.find("bndbox").find("xmax").text)
        ymax=np.float32(obj.find("bndbox").find("ymax").text)

        width,height=xmax-xmin,ymax-ymin

        anno.write(f"{img_name} {int(xmin)} {int(ymin)} {int(width)} {int(height)}\n")
        anno.flush()
anno.close()
import os
from PIL import Image
import numpy as np
import traceback

anno_src=r"F:\data\CelebA\Anno\list_bbox_celeba.txt"
landmark_src=r"F:\data\CelebA\Anno\list_landmarks_celeba.txt"
img_src=r"F:\data\CelebA\Img\img_celeba.7z\img_celeba"
save_path=r"data"


def compute_iou(box1, box2):
    '''
    两个框（二维）的 iou 计算

    注意：边框以左上为原点

    box:[x1,y2,x2,y2],依次为左上右下坐标
    '''
    h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
    inter = w * h
    union = area_box1 + area_box2 - inter
    iou = inter / union
    return iou

for face_size in [12,24,48]:
    print("imgsize",face_size)
    positive_image_dir=os.path.join(save_path,str(face_size),"positive")
    negative_image_dir=os.path.join(save_path,str(face_size),"negative")
    part_image_dir=os.path.join(save_path,str(face_size),"part")

    for dir_path in [positive_image_dir,negative_image_dir,part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    positive_anno_path=os.path.join(save_path,str(face_size),"positive.txt")
    negative_anno_path=os.path.join(save_path,str(face_size),"negative.txt")
    part_anno_path=os.path.join(save_path,str(face_size),"part.txt")

    positive_account=0
    negative_account=0
    part_account=0

    try:
        positive_anno_file=open(positive_anno_path,"w")
        negative_anno_file=open(negative_anno_path,"w")
        part_anno_file=open(part_anno_path,"w")

        bbox=open(anno_src,"r").readlines()
        landmark=open(landmark_src,"r").readlines()

        for i,(bbox_line,landmark_line) in enumerate(zip(bbox,landmark)):
            if i<2:
                continue
            try:
                strs=bbox_line.strip().split()#等同于readline
                img_name=strs[0].strip()
                # print(img_name)
                img_path=os.path.join(img_src,img_name)
                img=Image.open(img_path)
                img_w,img_h=img.size
                x1= float(strs[1].strip())
                y1=float(strs[2].strip())
                w=float(strs[3].strip())
                h=float(strs[4].strip())

                strs1 = landmark_line.strip().split()  # 等同于readline
                px1 = float(strs1[1].strip())
                py1 = float(strs1[2].strip())
                px2 = float(strs1[3].strip())
                py2 = float(strs1[4].strip())
                px3 = float(strs1[5].strip())
                py3 = float(strs1[6].strip())
                px4 = float(strs1[7].strip())
                py4 = float(strs1[8].strip())
                px5 = float(strs1[9].strip())
                py5 = float(strs1[10].strip())

                if max(w,h)<40 or x1<0 or y1<0 or w<0 or h<0:
                    continue

                #小平偏移坐标
                x1=int(x1+w*0.12)
                y1=int(y1+h*0.1)
                x2=int(x1+w*0.9)
                y2=int(y1+h*0.85)
                boxes=[x1,y1,x2,y2]

                cx=x1+w/2
                cy=y1+h/2

                for _ in range(5):
                    w_=np.random.randint(-w*0.2,w*0.2)
                    h_=np.random.randint(-h*0.2,h*0.2)
                    cx_=cx+w_
                    cy_=cy+h_

                    side_len=np.random.randint(int(min(w,h)*0.8),np.ceil(1.25*max(w,h)))
                    x1_=np.max(cx_-side_len/2,0)
                    y1_=np.max(cy_-side_len/2,0)
                    x2_=x1_+side_len
                    y2_=y1_+side_len

                    crop_box=np.array([x1_,y1_,x2_,y2_])

                    offset_x1=(x1-x1_)/side_len
                    offset_y1=(y1-y1_)/side_len
                    offset_x2=(x2-x2_)/side_len
                    offset_y2=(y2-y2_)/side_len

                    offset_px1 = (px1 - x1_) / side_len
                    offset_py1 = (py1 - y1_) / side_len
                    offset_px2 = (px2 - x1_) / side_len
                    offset_py2 = (py2 - y1_) / side_len
                    offset_px3 = (px3 - x1_) / side_len
                    offset_py3 = (py3 - y1_) / side_len
                    offset_px4 = (px4 - x1_) / side_len
                    offset_py4 = (py4 - y1_) / side_len
                    offset_px5 = (px5 - x1_) / side_len
                    offset_py5 = (py5 - y1_) / side_len

                    face_crop=img.crop(crop_box)
                    face_resize=face_crop.resize((face_size,face_size),Image.ANTIALIAS)

                    iou=compute_iou(crop_box,np.array(boxes))

                    if iou>0.6:
                        positive_anno_file.write(f"positive/{positive_account}.jpg 1 {offset_x1} {offset_y1} {offset_x2} {offset_y2} {offset_px1} {offset_py1} {offset_px2} {offset_py2} {offset_px3} {offset_py3} {offset_px4} {offset_py4} {offset_px5} {offset_py5}\n")
                        positive_anno_file.flush()
                        face_resize.save(os.path.join(positive_image_dir,f"{positive_account}.jpg"))
                        positive_account+=1

                    elif 0.6>iou>0.4:
                        part_anno_file.write(f"part/{part_account}.jpg 2 {offset_x1} {offset_y1} {offset_x2} {offset_y2} {offset_px1} {offset_py1} {offset_px2} {offset_py2} {offset_px3} {offset_py3} {offset_px4} {offset_py4} {offset_px5} {offset_py5}\n")
                        part_anno_file.flush()
                        face_resize.save(os.path.join(part_image_dir, f"{part_account}.jpg"))
                        part_account += 1
                    elif iou<0.1:
                        negative_anno_file.write(f"negative/{negative_account}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
                        negative_anno_file.flush()
                        face_resize.save(os.path.join(negative_image_dir, f"{negative_account}.jpg"))
                        negative_account += 1

                _boxes=np.array(boxes)
                for i in range(5):
                    side_len=np.random.randint(face_size,min(img_h,img_w)/2)
                    x_=np.random.randint(0,img_w-side_len)
                    y_=np.random.randint(0,img_h-side_len)
                    crop_box=np.array([x_,y_,x_+side_len,y_+side_len])
                    if np.max(compute_iou(crop_box,_boxes))<0.1:
                        face_crop=img.crop(crop_box)
                        face_resize=face_crop.resize((face_size,face_size),Image.ANTIALIAS)
                        negative_anno_file.write(f"negative/{negative_account}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
                        negative_anno_file.flush()
                        face_resize.save(os.path.join(negative_image_dir, f"{negative_account}.jpg"))
                        negative_account += 1
            except Exception as e:
                traceback.print_exc()
    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()



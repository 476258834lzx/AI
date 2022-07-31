import math
import random
import os
import glob
import cv2
import shutil
import numpy as np
from PIL import Image,ImageFilter

def check_pix_num(img,h,w):
    img=cv2.copyMakeBorder(img,0,4-h%4,0,4-w%4,cv2.BORDER_CONSTANT,value=0)
    return img

def Laplace_fusion(bg_img,material_img,mask):
    h,w=bg_img.shape[0:2]
    bg_img=check_pix_num(bg_img,h,w)
    material_img=check_pix_num(material_img,h,w)
    mask=check_pix_num(mask,h,w)
    half_mask=cv2.resize(mask,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
    lpM=[half_mask,mask]

    G = bg_img.copy()
    gpA = [G]
    for i in range(2):
        G = cv2.pyrDown(G)
        gpA.append(G)

    G = material_img.copy()
    gpB = [G]
    for i in range(2):
        G = cv2.pyrDown(G)
        gpB.append(G)

    lpA = [gpA[1]]
    for i in range(1, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        # print(GE.shape)
        # print(gpA[i-1].shape)
        L = cv2.subtract(gpA[i - 1], GE)  # 金字塔低一层是i+1，高一层是i-1
        lpA.append(L)

    lpB = [gpB[1]]
    for i in range(1, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    LS = []
    for i, (la, lb,lm) in enumerate(zip(lpA, lpB,lpM)):
        lh,lw=la.shape[0:2]
        black=np.zeros([lh,lw,3],dtype=np.uint8)
        black[:,:,0]=lm
        black[:,:,1]=lm
        black[:,:,2]=lm
        white=cv2.bitwise_not(black)
        la=cv2.bitwise_and(la,white,(lw,lh))
        lb=cv2.bitwise_and(lb,black,(lw,lh))

        ls =cv2.add(la,lb,(lw,lh))
        # ls=np.hstack((la[:,0:250],lb[:,250:]))
        LS.append(ls)

    ls = LS[0]
    for i in range(1, 2):
        ls_ = cv2.pyrUp(ls)
        ls_ = cv2.add(ls_, LS[i])

    outimg=ls_[0:h,0:w,:]


    return outimg

def check_path(dst_folder,state):
    dst_path = os.path.join(dst_folder, state)
    aug_img_path = os.path.join(dst_path, "images")
    aug_label_path = os.path.join(dst_path, "labels")
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    if not os.path.exists(aug_img_path):
        os.mkdir(aug_img_path)
    if not os.path.exists(aug_label_path):
        os.mkdir(aug_label_path)

    return aug_img_path,aug_label_path

def flip(img_path,label_path,dst_folder):
    state="flip"
    aug_img_path,aug_label_path=check_path(dst_folder,state)
    img=cv2.imread(img_path)
    aug_img=cv2.flip(img,1)
    cv2.imwrite(os.path.join(aug_img_path,os.path.basename(img_path)[:-4]+f"-{state}.jpg"),aug_img)
    label=open(label_path)
    dst_label=open(os.path.join(aug_label_path,os.path.basename(label_path)[:-4]+f"-{state}.txt"),"a")
    for line in label.readlines():
        strs=line.split()
        cls,_x,_y,_w,_h=strs[0:5]
        x=1-np.float32(_x)
        y=np.float32(_y)
        dst_label.write(f"{cls} {x} {y} {_w} {_h}\n")
        dst_label.flush()
    label.close()
    dst_label.close()
    return dst_folder

def letter_box(img):
    height,width=img.shape[0:2]
    if height>width:
        rate=float(640/height)
        resize_img=cv2.resize(img,(int(width*rate),640),interpolation=cv2.INTER_LANCZOS4)
        top=0
        bottom=0
        if (640-resize_img.shape[1])%2==0:
            left=(640-resize_img.shape[1])/2
            right=(640-resize_img.shape[1])/2
        else:
            left = (640 - resize_img.shape[1]) // 2
            right = (640 - resize_img.shape[1]) //2+1
        letter_img=cv2.copyMakeBorder(resize_img,int(top),int(bottom),int(left),int(right),cv2.BORDER_CONSTANT,value=0)
    else:
        rate = float(640/width)
        resize_img = cv2.resize(img, (640,int(height*rate)), interpolation=cv2.INTER_LANCZOS4)
        if (640 - resize_img.shape[0])%2==0:
            top = (640 - resize_img.shape[0]) / 2
            bottom = (640 - resize_img.shape[0]) / 2
        else:
            top = (640 - resize_img.shape[0]) // 2+1
            bottom = (640 - resize_img.shape[0]) // 2
        left = 0
        right = 0
        letter_img=cv2.copyMakeBorder(resize_img,int(top),int(bottom),int(left),int(right),cv2.BORDER_CONSTANT,value=0)

    return letter_img,top,bottom,left,right

def mixup(img_path,mix_img_path,label_path,mix_label_path,dst_folder):
    state = "mixup"
    aug_img_path,aug_label_path=check_path(dst_folder,state)
    alpha,belta=np.random.uniform(0.3,1,1),np.random.uniform(0.3,1,1)
    img=cv2.imread(img_path)
    ori_height,ori_width=img.shape[0:2]
    mix_img=cv2.imread(mix_img_path)
    mix_height,mix_width=mix_img.shape[0:2]

    img,ori_top,ori_bottom,ori_left,ori_right=letter_box(img)
    mix_img,mix_top,mix_bottom,mix_left,mix_right=letter_box(mix_img)
    aug_img=cv2.addWeighted(img,float(alpha),mix_img,float(belta),0)
    cv2.imwrite(os.path.join(aug_img_path,os.path.basename(img_path)[:-4]+f"-{state}.jpg"),aug_img)
    label = open(label_path)
    mix_label=open(mix_label_path)
    dst_label=open(os.path.join(aug_label_path,os.path.basename(label_path)[:-4]+f"-{state}.txt"),"a")
    for line in label.readlines():
        strs=line.split()
        cls,_x,_y,_w,_h=strs[0:5]
        x=(np.float32(_x)*ori_width+ori_left)/(ori_left+ori_width+ori_right)
        y=(np.float32(_y)*ori_height+ori_top)/(ori_top+ori_height+ori_bottom)
        w=np.float32(_w)*ori_width/(ori_left+ori_width+ori_right)
        h=np.float32(_h)*ori_height/(ori_top+ori_height+ori_bottom)

        dst_label.write(f"{cls} {x} {y} {w} {h}\n")
        dst_label.flush()
    for line in mix_label.readlines():
        strs=line.split()
        cls,_x,_y,_w,_h=strs[0:5]

        x = (np.float32(_x) * mix_width + mix_left) / (mix_left + mix_width + mix_right)
        y = (np.float32(_y) * mix_height + mix_top) / (mix_top + mix_height + mix_bottom)
        w = np.float32(_w) * mix_width / (mix_left + mix_width + mix_right)
        h = np.float32(_h) * mix_height / (mix_top + mix_height + mix_bottom)
        dst_label.write(f"{cls} {x} {y} {w} {h}\n")
        dst_label.flush()
    label.close()
    mix_label.close()
    dst_label.close()
    return dst_folder

def mosaic(img_path,label_path,dst_folder):
    state = "mosaic"
    aug_img_path,aug_label_path=check_path(dst_folder,state)

    label = open(label_path)
    img = cv2.imread(img_path)
    height,width=img.shape[0:2]
    for line in label.readlines():
        strs = line.split()
        cls, _x, _y, _w, _h = strs[0:5]
        x1 = int(np.around(np.float32(_x)*width-np.float32(_w)*width/2,decimals=0))
        y1 = int(np.around(np.float32(_y)*height-np.float32(_h)*height/2,decimals=0))
        x2 = int(np.around(np.float32(_x) * width + np.float32(_w) * width / 2, decimals=0))
        y2 = int(np.around(np.float32(_y) * height + np.float32(_h) * height / 2, decimals=0))
        w=int(np.around(np.float32(_w)*width,decimals=0))
        h=int(np.around(np.float32(_h)*height,decimals=0))

        mosaic_w=int(np.around(np.random.uniform(w*0.05,w*0.1,1),decimals=0))
        mosaic_h=int(np.around(np.random.uniform(h*0.05,h*0.1,1),decimals=0))

        num=random.randint(0,5)
        for i in range(num):
            mosaic_x1=int(np.around(np.random.uniform(x1,x2-mosaic_w,1),decimals=0))
            mosaic_y1=int(np.around(np.random.uniform(y1,y2-mosaic_w,1),decimals=0))
            mosaic_x2=mosaic_x1+mosaic_w
            mosaic_y2=mosaic_y1+mosaic_h
            cv2.rectangle(img,(mosaic_x1,mosaic_y1),(mosaic_x2,mosaic_y2),(0,0,0),-1)

    cv2.imwrite(os.path.join(aug_img_path, os.path.basename(img_path)[:-4] + f"-{state}.jpg"), img)
    shutil.copy(label_path,aug_label_path)
    mosaic_labelname=os.path.join(aug_label_path,os.path.basename(label_path))
    os.rename(mosaic_labelname,mosaic_labelname[:-4]+f"-{state}.txt")
    return dst_folder

def hsv(img_path,label_path,dst_folder):
    state = "hsv"
    aug_img_path,aug_label_path=check_path(dst_folder,state)

    img=cv2.imread(img_path)
    # r = np.random.uniform(-1, 1, 3) * [0.5, 0.5, 0.5] + 1  # random gains
    r = np.clip(np.random.normal(1, 1, 3), 0.5, 1.5)
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)#RGB增量转HSV
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    cv2.imwrite(os.path.join(aug_img_path, os.path.basename(img_path)[:-4] + f"-{state}.jpg"), img)
    shutil.copy(label_path, aug_label_path)
    mosaic_labelname = os.path.join(aug_label_path, os.path.basename(label_path))
    os.rename(mosaic_labelname, mosaic_labelname[:-4] + f"-{state}.txt")
    return dst_folder

def translate(img_path,label_path,dst_folder):
    state = "translate"
    aug_img_path,aug_label_path=check_path(dst_folder,state)

    img=cv2.imread(img_path)
    height,width=img.shape[0:2]
    X=np.around(np.clip(np.random.randn(1),-0.1,0.1)*width,decimals=0)
    Y=np.around(np.clip(np.random.randn(1),-0.1,0.1)*height,decimals=0)
    M=np.array([[1,0,int(X)],[0,1,int(Y)],[0,0,1]],dtype=float)
    aug_img=cv2.warpPerspective(img,M,(width,height))
    cv2.imwrite(os.path.join(aug_img_path, os.path.basename(img_path)[:-4] + f"-{state}.jpg"), aug_img)
    label=open(label_path)
    dst_label=open(os.path.join(aug_label_path,os.path.basename(label_path)[:-4]+f"-{state}.txt"),"a")
    for line in label.readlines():
        strs=line.split()
        cls,_x,_y,_w,_h=strs[0:5]

        x1=np.float32(_x)-np.float32(_w)/2
        y1=np.float32(_y)-np.float32(_h)/2
        x2=np.float32(_x)+np.float32(_w)/2
        y2=np.float32(_y)+np.float32(_h)/2

        new_x1=min(max(0,x1+X.squeeze(0)/width),1)
        new_y1=min(max(0,y1+Y.squeeze(0)/height),1)
        new_x2=min(max(0,x2+X.squeeze(0)/width),1)
        new_y2=min(max(0,y2+Y.squeeze(0)/height),1)

        w=abs(new_x2-new_x1)
        h=abs(new_y2-new_y1)
        x=(new_x1+new_x2)/2
        y=(new_y1+new_y2)/2

        dst_label.write(f"{cls} {x} {y} {w} {h}\n")
        dst_label.flush()
    label.close()
    dst_label.close()
    return dst_folder

def perspective(img_path,label_path,dst_folder):
    state = "perspective"
    aug_img_path,aug_label_path=check_path(dst_folder,state)

    img=cv2.imread(img_path)
    height,width=img.shape[0:2]
    # Perspective
    P = np.eye(3)
    # P[2, 0] = random.uniform(-0.001, 0.001)  # x perspective (about y)
    P[2, 0] = np.clip(np.random.randn(1),-0.0005,0.0005).squeeze(0)  # x perspective (about y)
    # P[2, 1] = random.uniform(-0.001, 0.001)  # y perspective (about x)
    P[2, 1] = np.clip(np.random.randn(1),-0.0005,0.0005).squeeze(0)  # y perspective (about x)
    aug_img = cv2.warpPerspective(img, P, dsize=(width, height))
    cv2.imwrite(os.path.join(aug_img_path, os.path.basename(img_path)[:-4] + f"-{state}.jpg"), aug_img)
    label=open(label_path)
    dst_label = open(os.path.join(aug_label_path, os.path.basename(label_path)[:-4] + f"-{state}.txt"), "a")
    for line in label.readlines():
        strs = line.split()
        cls, _x, _y, _w, _h = strs[0:5]

        x1 = (np.float32(_x) - np.float32(_w) / 2)*width
        y1 = (np.float32(_y) - np.float32(_h) / 2)*height
        x2 = (np.float32(_x) + np.float32(_w) / 2)*width
        y2 = (np.float32(_y) + np.float32(_h) / 2)*height

        if (x1)/(P[2,0]*x1+P[2,1]*y1+1)>width or (y1)/(P[2,0]*x1+P[2,1]*y1+1)>height or (x2)/(P[2,0]*x2+P[2,1]*y2+1)<0 or (y2)/(P[2,0]*x2+P[2,1]*y2+1)<0:
            continue

        new_x1 = max(0, (x1)/(P[2,0]*x1+P[2,1]*y1+1))
        new_y1 = max(0, (y1)/(P[2,0]*x1+P[2,1]*y1+1))
        new_x2 = min((x2)/(P[2,0]*x2+P[2,1]*y2+1), width)
        new_y2 = min((y2)/(P[2,0]*x2+P[2,1]*y2+1), height)

        w = abs(new_x2 - new_x1)/width
        h = abs(new_y2 - new_y1)/height
        x = (new_x1 + new_x2) / (2*width)
        y = (new_y1 + new_y2) / (2*height)

        dst_label.write(f"{cls} {x} {y} {w} {h}\n")
        dst_label.flush()
    label.close()
    dst_label.close()

    return dst_folder

def goal_mask_Laplace_fusion(img_path,label_path,material_src,mark,dst_folder,fusion_cls):
    state = "goal_mask_Laplace_fusion"
    aug_img_path,aug_label_path=check_path(dst_folder,state)

    label = open(label_path)
    dst_label = open(os.path.join(aug_label_path, os.path.basename(label_path)[:-4] + f"-{state}.txt"), "a")
    for line in label.readlines():
        strs=line.split()
        cls,_x,_y,_w,_h=strs[0:5]
        dst_label.write(f"{cls} {_x} {_y} {_w} {_h}\n")
        dst_label.flush()
    label.close()

    img = cv2.imread(img_path)
    bg_img=img.copy()
    bg_height,bg_width=bg_img.shape[0:2]

    roi_height,roi_width=mark[3]-mark[1],mark[2]-mark[0]
    origin_x,origin_y=mark[0:2]

    amount=np.random.randint(1,4)
    for i in range(amount):
        index = np.random.randint(0, len(glob.glob(material_src)))
        material_img_path = glob.glob(material_src)[index]
        material_img=cv2.imread(material_img_path,-1)
        m_height,m_width=material_img.shape[0:2]
        ratio=np.random.uniform(0.2,0.5,1).squeeze(0)

        if (roi_width/roi_height)>=(m_width/m_height):
            material_scale_height=int(roi_height*ratio)
            material_scale_width=int(roi_height*ratio/m_height*m_width)
        else:
            material_scale_width=int(roi_width*ratio)
            material_scale_height=int(roi_width*ratio/m_width*m_height)

        material_scale_img=cv2.resize(material_img,(material_scale_width,material_scale_height),interpolation=cv2.INTER_LINEAR)

        offset_x=int(np.around(np.random.uniform(0,roi_width-material_scale_width,1).squeeze(0),decimals=0))
        offset_y=int(np.around(np.random.uniform(0,roi_height-material_scale_height,1).squeeze(0),decimals=0))

        x1=origin_x+offset_x
        y1=origin_y+offset_y
        x2=x1+material_scale_width
        y2=y1+material_scale_height
        x = (x1 + x2) / 2 - 1
        y = (y1 + y2) / 2 - 1

        _x,_y,_w,_h=x/bg_width,y/bg_height,material_scale_width/bg_width,material_scale_height/bg_height

        dst_label.write(f"{fusion_cls} {_x} {_y} {_w} {_h}\n")
        dst_label.flush()

        left=offset_x+origin_x
        right=bg_width-left-material_scale_width
        top=offset_y+origin_y
        bottom=bg_height-top-material_scale_height

        mask=material_scale_img[:,:,-1]
        material_scale_bgr=material_scale_img[:,:,0:3]

        new_mask=cv2.copyMakeBorder(mask,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
        new_material_bgr=cv2.copyMakeBorder(material_scale_bgr,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
        bg_img=Laplace_fusion(bg_img,new_material_bgr,new_mask)
    cv2.imwrite(os.path.join(aug_img_path, os.path.basename(img_path)[:-4] + f"-{state}.jpg"), bg_img)
    dst_label.close()
    return dst_folder

def sharpening(img_path,label_path,dst_folder,if_Gauss=True):
    state = "sharpening"
    aug_img_path,aug_label_path=check_path(dst_folder,state)

    img=cv2.imread(img_path)

    if if_Gauss:
        aug_img = cv2.GaussianBlur(img, (3, 3), 0)
        aug_img = cv2.addWeighted(img, 2, aug_img, -1, 0)
    else:
        gradient=cv2.Laplacian(img,-1,(3,3))
        aug_img=cv2.subtract(img,gradient)

    cv2.imwrite(os.path.join(aug_img_path, os.path.basename(img_path)[:-4] + f"-{state}.jpg"), aug_img)
    shutil.copy(label_path, aug_label_path)
    mosaic_labelname = os.path.join(aug_label_path, os.path.basename(label_path))
    os.rename(mosaic_labelname, mosaic_labelname[:-4] + f"-{state}.txt")

    return dst_folder

def hist_equalist(img_path,label_path,dst_folder):
    state = "hist_equalist"
    aug_img_path,aug_label_path=check_path(dst_folder,state)

    img=cv2.imread(img_path)
    hsvimg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsvimg)
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v=clahe.apply(v)
    aug_img=cv2.cvtColor(cv2.merge([h,s,v]),cv2.COLOR_HSV2BGR)

    cv2.imwrite(os.path.join(aug_img_path, os.path.basename(img_path)[:-4] + f"-{state}.jpg"), aug_img)
    shutil.copy(label_path, aug_label_path)
    mosaic_labelname = os.path.join(aug_label_path, os.path.basename(label_path))
    os.rename(mosaic_labelname, mosaic_labelname[:-4] + f"-{state}.txt")
    return dst_folder

def get_contour(img_path,label_path,dst_folder):
    state = "get_contours"
    aug_img_path,aug_label_path=check_path(dst_folder,state)

    img=Image.open(img_path)
    aug_img = img.filter(ImageFilter.CONTOUR)

    aug_img.save(os.path.join(aug_img_path, os.path.basename(img_path)[:-4] + f"-{state}.jpg"))
    shutil.copy(label_path, aug_label_path)
    mosaic_labelname = os.path.join(aug_label_path, os.path.basename(label_path))
    os.rename(mosaic_labelname, mosaic_labelname[:-4] + f"-{state}.txt")

    return dst_folder

if __name__ == '__main__':
    material_src = "/home/yitutong/桌面/liuzhenxing/data/yolo/material/*.png"
    img_src = "/home/yitutong/桌面/liuzhenxing/data/yolo/images/*.jpg"
    label_src = "/home/yitutong/桌面/liuzhenxing/data/yolo/labels"
    dst_folder = "/home/yitutong/桌面/liuzhenxing/data/augument"
    for img_path in glob.glob(img_src):
        label_path=os.path.join(label_src,os.path.basename(img_path)[:-3]+"txt")
        # flip(img_path, label_path, dst_folder)#翻转增量
        # index=np.random.randint(0,len(glob.glob(img_src)))#mixup增量
        # mix_img_path=glob.glob(img_src)[index]
        # mix_label_path=os.path.join(label_src,os.path.basename(mix_img_path)[:-3]+"txt")
        # mixup(img_path,mix_img_path,label_path,mix_label_path,dst_folder)
        # mosaic(img_path,label_path,dst_folder)#mosaic增量
        # hsv(img_path,label_path,dst_folder)#hsv增量
        # translate(img_path,label_path,dst_folder)#平移增量
        # perspective(img_path,label_path,dst_folder)#透视增量(不适用于长条状检测目标)
        # mark=np.array([0,0,100,100],np.int32)#融合目标增量,x1,y1,x2,y2
        # goal_mask_Laplace_fusion(img_path,label_path,material_src,mark,dst_folder,0)
        # sharpening(img_path,label_path,dst_folder,True)#锐化图像增量
        # hist_equalist(img_path,label_path,dst_folder)#直方图均衡增量
        get_contour(img_path,label_path,dst_folder)#轮廓图增量
import torch
from PIL import Image,ImageDraw
import numpy as np
from utils import general
import net
from torchvision import transforms
import time
import os

p_cls=0.6
p_nms=0.5

r_cls=0.6
r_nms=0.5

o_cls=0.3
o_nms=0.5

class Detecter:
    def __init__(self,pnet_param,rnet_param,onet_param,isCuda=True):
        self.isCuda=isCuda
        self.pnet=pnet_param
        self.rnet=rnet_param
        self.onet=onet_param

        if self.isCuda:
            self.pnet = self.pnet.cuda()
            self.rnet = self.rnet.cuda()
            self.onet = self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__image__transformer=transforms.Compose([
            transforms.ToTensor()
        ])

    def detect(self,image):
        #P网络检测
        start_time=time.time()
        pnet_boxes=self.__pnet_detect(image)
        if pnet_boxes.shape[0]==0:
            return np.array([])
        end_time=time.time()
        p_time=end_time-start_time

        #R网络检测
        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image,pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        r_time = end_time - start_time

        #O网络检测
        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        o_time = end_time - start_time

        all_time=p_time+r_time+o_time
        print(f"time:{all_time} p:{p_time} r:{r_time} o:{o_time}")

        return onet_boxes

    def __box(self,start_index,offset,landmark,cls,scale,stride=2,sile_len=12):
        _x1=(start_index[1].float()*stride)/scale
        _y1=(start_index[0].float()*stride)/scale
        _x2=(start_index[1].float()*stride+sile_len-1)/scale
        _y2=(start_index[0].float()*stride+sile_len-1)/scale

        ow=_x2-_x1
        oh=_y2-_y1

        _offset=offset[:,start_index[0],start_index[1]]
        x1=_x1+ow*_offset[0]
        y1=_y1+oh*_offset[1]
        x2=_x2+ow*_offset[2]
        y2 = _y2 + oh * _offset[3]

        _landmark = landmark[:, start_index[0], start_index[1]]
        px1 = _x1+ow*_landmark[0]
        py1 = _y1+oh*_landmark[1]
        px2 = _x1+ow*_landmark[2]
        py2 = _y1+oh*_landmark[3]
        px3 = _x1+ow*_landmark[4]
        py3 = _y1+oh*_landmark[5]
        px4 = _x1+ow*_landmark[6]
        py4 = _y1+oh*_landmark[7]
        px5 = _x1+ow*_landmark[8]
        py5 = _y1+oh*_landmark[9]

        return [x1,y1,x2,y2,cls,px1,py1,px2,py2,px3,py3,px4,py4,px5,py5]


    def __pnet_detect(self,image):
        boxes=[]
        img=image
        w,h=img.size
        min_side_len=min(w,h)#最小边长

        scale=1
        while min_side_len>12:
            img_data=self.__image__transformer(img)
            if self.isCuda:
                img_data=img_data.cuda()

            img_data.unsqueeze_(0)

            _cls,_offset,_landmark=self.pnet(img_data)

            cls=_cls[0][0].cpu().data
            offset=_offset[0].cpu().data
            landmark=_landmark[0].cpu().data
            idxs=torch.nonzero(torch.gt(cls,p_cls))#HW中的True索引

            for idx in idxs:
                boxes.append(self.__box(idx,offset,landmark,cls[idx[0],idx[1]],scale))

            scale*=0.7
            _w,_h=int(w*scale),int(h*scale)

            img=img.resize((_w,_h))
            min_side_len=min(_w,_h)

        return general.nms(torch.tensor(boxes),p_nms)





if __name__ == '__main__':
    image_path=r"test_img"

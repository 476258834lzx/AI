import torch
from PIL import Image,ImageDraw
from utils import general
from net import *
from torchvision import transforms
import time
import os

p_cls=0.6
p_nms=0.5

r_cls=0.6
r_nms=0.5

o_cls=0.3
o_nms=0.5

def square(boxes):
    square_bbox=boxes
    if boxes.shape[0]==0:
        return torch.tensor([])
    h=boxes[:,3]-boxes[:,1]
    w=boxes[:,2]-boxes[:,0]
    max_side=torch.maximum(h,w)
    square_bbox[:,0]=boxes[:,0]+w*0.5-max_side*0.5
    square_bbox[:,1]=boxes[:,1]+h*0.5-max_side*0.5
    square_bbox[:,2]=square_bbox[:,0]+max_side
    square_bbox[:,3]=square_bbox[:,1]+max_side
    return square_bbox

class Detecter:
    def __init__(self,pnet_param,rnet_param,onet_param,isCuda=True):
        self.isCuda=isCuda
        self.pnet=PNet()
        self.rnet=RNet()
        self.onet=ONet()

        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        if self.isCuda:
            self.pnet = self.pnet.cuda()
            self.rnet = self.rnet.cuda()
            self.onet = self.onet.cuda()

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
            return torch.tensor([])
        end_time=time.time()
        p_time=end_time-start_time

        #R网络检测
        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image,pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return torch.tensor([])
        end_time = time.time()
        r_time = end_time - start_time

        #O网络检测
        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return torch.tensor([])
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
            img_data=img_data[:3,...]
            if self.isCuda:
                img_data=img_data.cuda()

            img_data.unsqueeze_(0)

            _cls,_offset,_landmark=self.pnet(img_data)

            cls=_cls[0][0]
            offset=_offset[0]
            landmark=_landmark[0]
            idxs=torch.nonzero(torch.gt(cls,p_cls))#HW中的True索引

            for idx in idxs:
                boxes.append(self.__box(idx,offset,landmark,cls[idx[0],idx[1]],scale))

            scale*=0.7
            _w,_h=int(w*scale),int(h*scale)

            img=img.resize((_w,_h))
            min_side_len=min(_w,_h)

        return general.nms(torch.tensor(boxes),p_nms)

    def __rnet_detect(self,image,pnet_boxes):
        _img_dataset=[]
        _pnet_boxes=square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1=int(_box[0])
            _y1=int(_box[1])
            _x2=int(_box[2])
            _y2=int(_box[3])
            img=image.crop((_x1,_y1,_x2,_y2))
            img=img.resize((24,24))
            img_data=self.__image__transformer(img)
            img_data = img_data[:3, ...]
            _img_dataset.append(img_data)
        img_dataset=torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset=img_dataset.cuda()

        #N1    N4      N10
        _cls,_offset,_landmark=self.rnet(img_dataset)

        boxes=[]
        idxs,_=torch.where(_cls>r_cls)#不是N*1,N*1是逻辑运算得到的布尔矩阵的形状，这里直接拿到索引
        for idx in idxs:#1,True or False
            _box=_pnet_boxes[idx]
            _x1=int(_box[0])
            _y1=int(_box[1])
            _x2=int(_box[2])
            _y2=int(_box[3])

            ow=_x2-_x1
            oh=_y2-_y1

            x1 = _x1 + ow * _offset[idx][0]
            y1 = _y1 + oh * _offset[idx][1]
            x2 = _x2 + ow * _offset[idx][2]
            y2 = _y2 + oh * _offset[idx][3]
            px1 = _x1 + ow * _landmark[idx][0]
            py1 = _y1 + oh * _landmark[idx][1]
            px2 = _x1 + ow * _landmark[idx][2]
            py2 = _y1 + oh * _landmark[idx][3]
            px3 = _x1 + ow * _landmark[idx][4]
            py3 = _y1 + oh * _landmark[idx][5]
            px4 = _x1 + ow * _landmark[idx][6]
            py4 = _y1 + oh * _landmark[idx][7]
            px5 = _x1 + ow * _landmark[idx][8]
            py5 = _y1 + oh * _landmark[idx][9]

            boxes.append([x1,y1,x2,y2,_cls[idx][0],px1,py1,px2,py2,px3,py3,px4,py4,px5,py5])

        return  general.nms(torch.tensor(boxes),r_nms)

    def __onet_detect(self,image,rnet_boxes):
        _img_dataset = []
        _rnet_boxes = square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image__transformer(img)
            img_data = img_data[:3, ...]
            _img_dataset.append(img_data)
        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        # N1    N4      N10
        _cls, _offset, _landmark = self.onet(img_dataset)
        
        boxes = []
        idxs, _ = torch.where(_cls > o_cls)  # 不是N*1,N*1是逻辑运算得到的布尔矩阵的形状，这里直接拿到索引
        for idx in idxs:  # 1,True or False
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * _offset[idx][0]
            y1 = _y1 + oh * _offset[idx][1]
            x2 = _x2 + ow * _offset[idx][2]
            y2 = _y2 + oh * _offset[idx][3]
            px1 = _x1 + ow * _landmark[idx][0]
            py1 = _y1 + oh * _landmark[idx][1]
            px2 = _x1 + ow * _landmark[idx][2]
            py2 = _y1 + oh * _landmark[idx][3]
            px3 = _x1 + ow * _landmark[idx][4]
            py3 = _y1 + oh * _landmark[idx][5]
            px4 = _x1 + ow * _landmark[idx][6]
            py4 = _y1 + oh * _landmark[idx][7]
            px5 = _x1 + ow * _landmark[idx][8]
            py5 = _y1 + oh * _landmark[idx][9]

            boxes.append([x1, y1, x2, y2, _cls[idx][0], px1, py1, px2, py2, px3, py3, px4, py4, px5, py5])

        return general.nms(torch.tensor(boxes), o_nms,isMin=True)


if __name__ == '__main__':
    image_path=r"test_img"
    for i in os.listdir(image_path):
        detector=Detecter(r"params\pnet.pt",r"params\rnet.pt",r"params\onet.pt")
        img=Image.open(os.path.join(image_path,i))
        boxes=detector.detect(img)
        imDraw=ImageDraw.Draw(img)
        for box in boxes:
            x1=int(box[0])
            y1=int(box[1])
            x2=int(box[2])
            y2=int(box[3])

            px1=int(box[5])
            py1=int(box[6])
            px2=int(box[7])
            py2=int(box[8])
            px3=int(box[9])
            py3=int(box[10])
            px4=int(box[11])
            py4=int(box[12])
            px5=int(box[13])
            py5=int(box[14])

            print((x1,y1,x2,y2))
            print((px1,py1,px2,py2,px3,py3,px4,py4,px5,py5))

            print("conf:",box[4])
            imDraw.rectangle((x1,y1,x2,y2),outline="red")
            imDraw.ellipse((px1-2,py1-2,px1+2,py1+2),outline="blue")
            imDraw.ellipse((px2-2,py2-2,px2+2,py2+2),outline="blue")
            imDraw.ellipse((px3-2,py3-2,px3+2,py3+2),outline="blue")
            imDraw.ellipse((px4-2,py4-2,px4+2,py4+2),outline="blue")
            imDraw.ellipse((px5-2,py5-2,px5+2,py5+2),outline="blue")

        img.show()
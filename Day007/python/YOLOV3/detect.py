import torch
import cv2
import os
from net import Yolov3
import cfg
from torch import nn
from YOLOV3.utils.general import *

classes=['vehicle','cons','person','parking_lock','column']

class Detecter:
    def __init__(self, net_param, anchors,class_num,isCuda=True):
        self.isCuda = isCuda
        self.anchors = anchors
        self.class_num = class_num
        self.net = Yolov3()

        self.net.load_state_dict(torch.load(net_param))

        if self.isCuda:
            self.net = self.net.cuda()

        self.net.eval()

    def detect(self, image,iou_thresh,nms_thresh):
        anchors_13 = torch.Tensor(self.anchors[13])
        anchors_26 = torch.Tensor(self.anchors[26])
        anchors_52 = torch.Tensor(self.anchors[52])
        if self.isCuda:
            image = image.cuda()
            anchors_13=anchors_13.cuda()
            anchors_26=anchors_26.cuda()
            anchors_52=anchors_52.cuda()

        output13, output26, output52 = self.net(image)
        idxs_13, vecs_13 = self._filter(output13, iou_thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors_13)

        idxs_26, vecs_26 = self._filter(output26, iou_thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors_26)

        idxs_52, vecs_52 = self._filter(output52, iou_thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors_52)
        all_boxes = torch.cat([boxes_13, boxes_26, boxes_52], dim=0).cpu()#N*6
        boxes=[]
        for i in range(self.class_num):
            mask=all_boxes[:,5]==i
            boxes.append(nms(all_boxes[mask],nms_thresh))

        return torch.cat(boxes, dim=0)

    def _filter(self,output,thresh):
        output = output.permute(0, 2, 3, 1)  # N,30,13,13->N,13,13,30
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)  # N,13,13,30->N,13,13,3,10
        mask = torch.gt(output[..., 0],thresh)

        idxs=torch.nonzero(mask)#取索引N*4,此处N为目标数
        vecs=output[mask]#N*10,此处N为目标数

        return idxs,vecs

    def _parse(self,idxs,vecs,sampling_multiple,anchor):
        anchor=torch.Tensor(anchor)

        n=idxs[:,0]#索引第一个维度为目标数，第二个维度0:3，N*13*13*3第一个维度N为第几张的图
        a=idxs[:,3]#建议框

        cy=(idxs[:,1].float()+vecs[:,2])*sampling_multiple
        cx=(idxs[:,2].float()+vecs[:,1])*sampling_multiple

        w=anchor[a,0]*torch.exp(vecs[:,3])
        h=anchor[a,1]*torch.exp(vecs[:,4])
        conf=vecs[:,0]
        cls=torch.argmax(vecs[:,5:10],dim=1)
                            #n为batch
        # return torch.stack([n.float(),conf,cx,cy,w,h,cls],dim=1)
        return torch.stack([conf,cx,cy,w,h,cls],dim=1)


if __name__ == '__main__':
    image_path = r"test_img"
    color = np.random.randint(0, 256, (len(classes), 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    detector = Detecter(r"weights\best.pt", cfg.ANCHORS, cfg.CLASS_NUM)
    for i in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, i))
        img_data, top, _, left, _, rate=letter_box(img,416)
        img_data = torch.Tensor((img_data / 255 - 0.5).transpose(2, 0, 1)).unsqueeze_(0)
        boxes = detector.detect(img_data,0.25,0.45).detach().numpy()
        for goal in boxes:
            conf,cx,cy,w,h,cls=goal[0:6]
            revise_cx,revise_cy=(cx-left)/rate,(cy-top)/rate
            revise_w,revise_h=w/rate,h/rate
            x1,y1,x2,y2=int(revise_cx-revise_w/2),int(revise_cy-revise_h/2),int(revise_cx+revise_w/2),int(revise_cy+revise_h/2)
            # print(conf,revise_cx,revise_cy,revise_w,revise_h,cls)
            cls_color=color[int(cls)].tolist()
            cv2.rectangle(img,(x1,y1),(x2,y2),color=cls_color,thickness=2)#左上角，右下角，颜色，线宽-1填充
            cv2.putText(img, f"{classes[int(cls)]}:{conf}", (x1,y1-10), font,1,cls_color, 1,
                        lineType=cv2.LINE_AA)  # 字符，字符左上角，字体，字体间距，颜色，线宽，像素补值法防锯齿
            #
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
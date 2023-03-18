import torch
import numpy as np
import cv2

def iou(box,boxes,isMin=False,isTorch=True):
    # conf,cx,cy,w,h,cls
    if isTorch:
        box_area = box[3]*box[4]
        area = boxes[:,3]*boxes[:,4]

        xx1 = torch.maximum(box[1]-box[3]/2, boxes[:,1]-boxes[:,3]/2)
        yy1 = torch.maximum(box[2]-box[4]/2, boxes[:,2]-boxes[:,4]/2)
        xx2 = torch.minimum(box[1]+box[3]/2, boxes[:,1]+boxes[:,3]/2)
        yy2 = torch.minimum(box[2]+box[4]/2, boxes[:,2]+boxes[:,4]/2)

        w = torch.maximum(torch.Tensor([0]), xx2 - xx1)
        h = torch.maximum(torch.Tensor([0]), yy2 - yy1)

        inter = w * h

        if isMin:
            ovr = torch.true_divide(inter, torch.minimum(box_area, area))
        else:
            ovr = torch.true_divide(inter, (box_area + area - inter))

    else:
        box_area = box[3] * box[4]
        area = boxes[:, 3] * boxes[:, 4]

        xx1 = np.maximum(box[1]-box[3]/2, boxes[:,1]-boxes[:,3]/2)
        yy1 = np.maximum(box[2]-box[4]/2, boxes[:,2]-boxes[:,4]/2)
        xx2 = np.minimum(box[1]+box[3]/2, boxes[:,1]+boxes[:,3]/2)
        yy2 = np.minimum(box[2]+box[4]/2, boxes[:,2]+boxes[:,4]/2)

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h

        if isMin:
            ovr = np.true_divide(inter, np.minimum(box_area, area))
        else:
            ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr

def nms(boxes,thresh,isMin=False,isTorch=True):
    if isTorch:
        if len(boxes)==0:
            return torch.Tensor([])
        #conf,cx,cy,w,h,cls
        _boxes=boxes[torch.argsort(-boxes[:,0])]
        r_boxes=[]

        while _boxes.shape[0]>1:
            a_box = _boxes[0]
            b_boxes = _boxes[1:]
            r_boxes.append(a_box)

            index = torch.where(iou(a_box, b_boxes, isMin, isTorch) < thresh)
            _boxes = b_boxes[index]

        if _boxes.shape[0]>0:
            r_boxes.append(_boxes[0])
    else:
        if len(boxes)==0:
            return np.array([])
        #x1,y1,x2,y2,c
        _boxes=boxes[np.argsort(-boxes[:,0])]
        r_boxes=[]

        while _boxes.shape[0]>1:
            a_box = _boxes[0]
            b_boxes = _boxes[1:]
            r_boxes.append(a_box)

            index = np.where(iou(a_box, b_boxes, isMin, isTorch) < thresh)
            _boxes = b_boxes[index]

        if _boxes.shape[0]>0:
            r_boxes.append(_boxes[0])

    return torch.stack(r_boxes) if isTorch else np.stack(r_boxes)

def convert_square(boxes):
    if boxes.shape[0] == 0:
        return np.array([])
    center_x = (boxes[:, 2] + boxes[:, 0]) / 2
    center_y = (boxes[:, 3] + boxes[:, 1]) / 2
    side = np.maximum((boxes[:, 2] - boxes[:, 0]), (boxes[:, 3] - boxes[:, 1]))
    boxes[:, 0] = center_x - side / 2
    boxes[:, 1] = center_y - side / 2
    boxes[:, 2] = center_x + side / 2
    boxes[:, 3] = center_y + side / 2
    # boxes = boxes.astype(np.int32)  #0.98的置信度会变成0

    boxes = boxes[boxes[:, 0] > 0]
    boxes = boxes[boxes[:, 1] > 0]
    boxes = boxes[boxes[:, 2] > 0]
    boxes = boxes[boxes[:, 3] > 0]
    return boxes

def letter_box(img,size=640):
    height,width=img.shape[0:2]
    if height>width:
        rate=float(size/height)
        resize_img=cv2.resize(img,(int(width*rate),size),interpolation=cv2.INTER_LANCZOS4)
        top=0
        bottom=0
        if (size-resize_img.shape[1])%2==0:
            left=(size-resize_img.shape[1])/2
            right=(size-resize_img.shape[1])/2
        else:
            left = (size - resize_img.shape[1]) // 2
            right = (size - resize_img.shape[1]) //2+1
        letter_img=cv2.copyMakeBorder(resize_img,int(top),int(bottom),int(left),int(right),cv2.BORDER_CONSTANT,value=0)
    else:
        rate = float(size/width)
        resize_img = cv2.resize(img, (size,int(height*rate)), interpolation=cv2.INTER_LANCZOS4)
        if (size - resize_img.shape[0])%2==0:
            top = (size - resize_img.shape[0]) / 2
            bottom = (size - resize_img.shape[0]) / 2
        else:
            top = (size - resize_img.shape[0]) // 2+1
            bottom = (size - resize_img.shape[0]) // 2
        left = 0
        right = 0
        letter_img=cv2.copyMakeBorder(resize_img,int(top),int(bottom),int(left),int(right),cv2.BORDER_CONSTANT,value=0)

    return letter_img,top,bottom,left,right,rate

if __name__ == '__main__':
    bs=np.random.randint(0,50,(10,14))
    print(bs)
    print(nms(bs,0.3,isTorch=False))
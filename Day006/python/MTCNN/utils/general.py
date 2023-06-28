import torch
import numpy as np

def iou(box,boxes,isMin=False,isTorch=True):

    if isTorch:
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        xx1 = torch.maximum(box[0], boxes[:,0])
        yy1 = torch.maximum(box[1], boxes[:,1])
        xx2 = torch.minimum(box[2], boxes[:,2])
        yy2 = torch.minimum(box[3], boxes[:,3])

        w = torch.maximum(torch.Tensor([0]), xx2 - xx1)
        h = torch.maximum(torch.Tensor([0]), yy2 - yy1)

        inter = w * h

        if isMin:
            ovr = torch.true_divide(inter, torch.minimum(box_area, area))
        else:
            ovr = torch.true_divide(inter, (box_area + area - inter))

    else:
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        xx1 = np.maximum(box[0], boxes[:,0])
        yy1 = np.maximum(box[1], boxes[:,1])
        xx2 = np.minimum(box[2], boxes[:,2])
        yy2 = np.minimum(box[3], boxes[:,3])

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
        #x1,y1,x2,y2,c
        _boxes=boxes[torch.argsort(-boxes[:,4])]
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
        _boxes=boxes[np.argsort(-boxes[:,4])]
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

if __name__ == '__main__':
    bs=np.random.randint(0,50,(10,14))
    print(bs)
    print(nms(bs,0.3,isTorch=False))
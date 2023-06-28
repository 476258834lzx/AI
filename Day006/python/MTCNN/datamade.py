from detect import *
from utils.general import *
class datamade:
    def __init__(self):
        self.savepath="face_recog/imgs"
        self.picpath="face_recog/temp"
        self.detector=Detecter(r"params\pnet.pt",r"params\rnet.pt",r"params\onet.pt")

    def run(self):
        positive_count=0
        for path in os.listdir(self.picpath):
            img_path = os.path.join(self.picpath, path)
            img = Image.open(img_path)
            # print(img.shape)
            boxes = self.detector.detect(img)
            boxes=convert_square(boxes)
            if boxes.shape[0] == 0:
                print("No face found")
                continue
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                confid = box[4]
                print("confidence:", confid)
            clip_img = img.crop([x1,y1,x2,y2])
            clip_img.save(f"{self.savepath}/{positive_count}.jpg")
            positive_count+=1
            print(positive_count)

if __name__ == '__main__':
    data=datamade()
    data.run()
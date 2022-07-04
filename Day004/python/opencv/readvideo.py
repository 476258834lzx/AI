import cv2

# cap=cv2.VideoCapture(0)
cap =cv2.VideoCapture("../img/1.MP4")
while True:
    ret,frame=cap.read()
    cv2.imshow('frame',frame)

    if cv2.waitKey(1)&0XFF==ord('q'):#防止有的操作系统waitkey返回值大于8位
        break

cap.release()
cv2.destroyAllWindows()
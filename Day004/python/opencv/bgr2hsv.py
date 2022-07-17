import cv2 as cv
import numpy as np
green = np.uint8([[[53,83,162]]])
print(green.shape)
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
print(hsv_green)
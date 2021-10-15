import numpy as np
from pylibdmtx.pylibdmtx import decode
import cv2
import numpy as np

image  = cv2.imread('Img_Data/23Fe9.png')
height, width = image.shape[:2]
msg = decode((image.tobytes(), width, height))
print(msg)


# image1 = cv2.imread('ROI')
# gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
# ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
#
# msg = pylibdmtx.decode(thresh1)
# print(msg)


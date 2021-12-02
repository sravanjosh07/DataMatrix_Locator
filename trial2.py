import cv2
import imutils
import numpy as np
from pylibdmtx.pylibdmtx import decode
import os

mycode_list = []

def preprocessed(frame):

    width = frame.shape[1]
    height = frame.shape[0]
    frame[500:height-500, :] = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

path = 'Img_Data'
output_path  = 'metrics.csv'

for filename in os.listdir(path):
    if filename.endswith('.jpg'):
        input_path = os.path.join(path, filename)

        frame = cv2.imread(input_path)
        frame = preprocessed(frame)

        th0 = frame
        th2 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        th3 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        ret, th1 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

        ret, th4 = cv2.threshold(frame, 60, 255, cv2.THRESH_OTSU)

        list1 = [th0, th1, th2, th3, th4]

        for img in list1:
            for code in decode(img):
                mycode = code.data.decode("utf-8")
                if mycode not in mycode_list:
                    print(mycode)
                with open('metrics.csv', 'a') as csv:
                    if mycode is not None:
                        csv.write('{},{}\n'.format(filename, 1))
                        continue
                    else:
                        csv.write('{},{}\n'.format(filename, 0))
                        continue
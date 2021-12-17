import cv2
import imutils
import numpy as np
from pylibdmtx.pylibdmtx import decode
import os
mycode_list = []
path = 'Img_Data'
output_path  = 'metrics.csv'
border_width = 500
def preprocessed(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def gradient_operation(frame):
    kernelx = np.array([[-1, 0, +1],
                        [-2, 0, +2],
                        [-1, 0, +1]])
    gradX = cv2.filter2D(frame, -1, kernelx)

    kernely = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [+1, +2, -1]])
    gradY = cv2.filter2D(frame, -1, kernely)

    # subtract the y-gradient from the x-gradient to learn
    gradient = cv2.subtract(gradX, gradY)
    # gradient = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
    gradient = cv2.convertScaleAbs(gradient)

    return gradient


def blur_and_morph(image):
    blurred = cv2.blur(image, (7, 7))
    (_, thresh1) = cv2.threshold(blurred, 20, 180, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 35))
    closed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=2)
    return closed


for filename in os.listdir(path):
    if filename.endswith('.jpg'):
        input_path = os.path.join(path, filename)

        frame = cv2.imread(input_path)
        gray = preprocessed(frame)
        gradient = gradient_operation(gray)
        closed = blur_and_morph(gradient)

        cnts, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.1 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            shape = np.shape(approx)
            if area > 30000 and shape[0] > 3:
                (x, y, w, h) = cv2.boundingRect(cnt)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                ROI = gray[y - 150:y + h + 50, x - 150:x + w + 100]
                print(area)
                if ROI is None:
                    pass

                else:
                    th0 = ROI
                    th2 = cv2.adaptiveThreshold(ROI, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
                    th3 = cv2.adaptiveThreshold(ROI, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                cv2.THRESH_BINARY, 11, 2)
                    ret, th1 = cv2.threshold(ROI, 127, 255, cv2.THRESH_BINARY)

                    ret, th4 = cv2.threshold(ROI, 60, 255, cv2.THRESH_OTSU)

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

cv2.namedWindow('namedWindow', cv2.WINDOW_NORMAL)
cv2.imshow('namedWindow', ROI)
cv2.waitKey(0)
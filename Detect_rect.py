import csv
import os

import cv2
import imutils
import numpy as np
from pylibdmtx import pylibdmtx

path = "img_data"
output_path = "metrics.csv"
dictionary_of_Boolean = {}

for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(path, filename)
        dictionary_of_Boolean[filename] = 0
        # load the image and convert it to grayscale
        image = cv2.imread(input_path)
        print(filename)
        assert image is not None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # invert_img = cv2.bitwise_not(gray)
        # ret, gray =cv2.threshold(invert_img, 0,255, cv2.THRESH_OTSU)
        # compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction using OpenCV 2.4

        # finding horizantal lines and vertical lines

        kernelx = np.array([[-1, 0, +1],
                            [-2, 0, +2],
                            [-1, 0, +1]])
        gradX = cv2.filter2D(gray, -1, kernelx)

        kernely = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [+1, +2, -1]])
        gradY = cv2.filter2D(gray, -1, kernely)

        # subtract the y-gradient from the x-gradient to learn
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # blurring
        blur = cv2.bilateralFilter(gradient, 9, 75, 75)

        # ret3, th5 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 3, -2)

        # list1 = [thresh4, thresh5, thresh3, thresh2, thresh1,th4, th5]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 15))
        closed = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=2)

        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=3)

        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=3)
        closed = cv2.dilate(closed, None, iterations=8)
        
        def additional_thresholding(roi):
            gray2 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            _, thresh2 = cv2.threshold(gray2, 110, 255, cv2.THRESH_BINARY)
            closed2 = cv2.erode(thresh2, None, iterations=2)
            closed2 = cv2.dilate(closed2, None, iterations=2)
            closed2 = cv2.erode(closed2, None, iterations=3)
            closed2 = cv2.dilate(closed2, None, iterations=3)

            closed2 = cv2.erode(closed2, None, iterations=4)
            closed2 = cv2.dilate(closed2, None, iterations=4)

            closed2 = cv2.erode(closed2, None, iterations=5)
            closed2 = cv2.dilate(closed2, None, iterations=5)
            return closed2

        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            rect = cv2.boundingRect(c)
            if 150 < rect[2] < 300 and 150 < rect[3] < 300:
                print(rect)

                print(cv2.contourArea(c))
                x, y, w, h = rect
                x1 = x - 75
                y1 = y - 75

                x1 = 0 if x1 < 0 else x1
                y1 = 0 if y1 < 0 else y1

                ROI = image[y1:y + h + 75, x1:x + w + 75]

                gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                # ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                thresh1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 97, 43)
                kernel = np.ones((5, 5), np.uint8)
                closed1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
                closed1 = cv2.erode(thresh1, None, iterations=3)
                closed1 = cv2.dilate(closed1, None, iterations=1)
                closed1 = cv2.erode(closed1, None, iterations=2)
                closed1 = cv2.dilate(closed1, None, iterations=2)

                # closed2 = additional_thresholding(ROI)
                msg1 = pylibdmtx.decode(closed1)
                # msg2 = pylibdmtx.decode(closed2)
                # msg = msg2 if not msg1 else msg1
                if msg1:
                    print(msg1)
                    dictionary_of_Boolean[filename] = 1
                    break
    with open('metrics2.csv', "a") as csv:

        csv.write("{},{}\n".format(filename, dictionary_of_Boolean[filename]))
        csv.flush()





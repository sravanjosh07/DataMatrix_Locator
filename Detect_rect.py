import csv
import os

import cv2
import imutils
import numpy as np
from pylibdmtx import pylibdmtx

path = "Img_Data"
output_path = "metrics.csv"
dictionary_of_Boolean = {}

for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(path,filename)

    # load the image and convert it to grayscale
        image = cv2.imread(input_path)
        print(filename)
        assert image is not None
        y = image.shape[0]
        x = image.shape[1]
        image[500:y-500, :] = 0
        # if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # invert_img = cv2.bitwise_not(gray)
        # ret, gray =cv2.threshold(invert_img, 0,255, cv2.THRESH_OTSU)
        # compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction using OpenCV 2.4

        #finding horizantal lines and vertical lines

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

        #blurring
        blur = cv2.bilateralFilter(gradient, 9, 75, 75)

        ret3, th5 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 3, -2)

        # list1 = [thresh4, thresh5, thresh3, thresh2, thresh1,th4, th5]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 15))
        closed = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=2)

        # cnts = imutils.grab_contours(cnts)

        # with open('metrics.csv', "a") as csv:
        #     csv.write("{},{}\n".format(filename, "not found"))

        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            rect = cv2.boundingRect(c)
            if rect[2] > 150 and rect[3] > 150:
                print(rect)

                print(cv2.contourArea(c))
                x, y, w, h = rect
                x1 = x - 50
                y1 = y - 50
                
                x1 = 0 if x1 < 0 else x1
                y1 = 0 if y1 < 0 else y1
                
                # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                # cv2.putText(image,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
                ROI = image[y1:y + h + 50, x1:x + w + 50]

                # image1 = cv2.imread('ROI')
                # if ROI is not None:
                gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                closed1 = cv2.erode(thresh1, None, iterations=2)
                closed1 = cv2.dilate(closed1, None, iterations=2)
                msg = pylibdmtx.decode(closed1)
                print(msg)
                with open('metrics.csv', "a") as csv:
                    if msg != []:
                            dictionary_of_Boolean[filename] = 1
                            csv.write("{},{}\n".format(filename, 1))
                            csv.flush()
                            break
                    else:
                            if filename not in dictionary_of_Boolean:
                                dictionary_of_Boolean[filename] = 0
                                csv.write("{},{}\n".format(filename, 0))
                                csv.flush()
                                continue
                    # else:
                    #     continue
                # else:
                #
                #     with open('metrics.csv', "a") as csv:
                #         csv.write("{},{}\n".format(filename, 2))
                #         csv.flush()
                #         continue


import csv
import os

import cv2
import imutils
import numpy as np
from pylibdmtx import pylibdmtx

path = "Img_Data"
output_path = "metrics.csv"
contour_offset = 50

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
        blurred = cv2.GaussianBlur(gradient,(7,11),0)
        ret3, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        #                             cv2.THRESH_BINARY, 3, -2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 25))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        closed = cv2.erode(morphed, None, iterations = 2)
        closed = cv2.dilate(closed, None, iterations = 2)

        cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)

        # with open('metrics.csv', "a") as csv:
        #     csv.write("{},{}\n".format(filename, "not found"))

        for cnt in cnts:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)  # approx is a three dimensional array of vertices
            shape = np.shape(approx)
            # finding area of the contour so we can consider contours having area greater than 7000 sq.pixels
            area = cv2.contourArea(cnt)
            if area >=10000 and area <= 45550 and shape[0] == 4:
                rect = cv2.minAreaRect(cnt)
                # box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
                # box = np.int0(box)
                # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(cnt)

                x1 = x - contour_offset
                y1 = y - contour_offset
                h1 = h + contour_offset
                w1 = w + contour_offset
                x1 = 0 if x1 < 0 else x1
                y1 = 0 if y1 < 0 else y1

                # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                # cv2.putText(image,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
                ROI = image[y1:y + h1, x1:x + w1]

                print(x,y,w,h)

                print(area)
                # image1 = cv2.imread('ROI')
                # if ROI is not None:
                gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                closed1 = cv2.erode(thresh1, None, iterations=2)
                closed1 = cv2.dilate(closed1, None, iterations=2)


                msg = pylibdmtx.decode(closed1)
                with open('metrics.csv', "a") as csv:
                    if msg == []:
                        csv.write("{},{}\n".format(filename, 0))
                        csv.flush()
                        continue
                    else:
                        csv.write("{},{}\n".format(filename, 1))
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


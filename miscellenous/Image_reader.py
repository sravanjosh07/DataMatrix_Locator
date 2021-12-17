import cv2
import imutils
import numpy as np
from pylibdmtx import pylibdmtx

#
# load the image and convert it to grayscale
image = cv2.imread(r"C:\Users\opv-operator\Desktop\DataMatrix_Locator-master\img_data\DetectTask 054804 BOTTOM 0.jpg")

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

# ret3, th5 = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                            cv2.THRESH_BINARY, 3, -2)
# th6 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)

# list1 = [thresh4, thresh5, thresh3, thresh2, thresh1,th4, th5]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 15))
closed = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

closed = cv2.erode(closed, None, iterations=2)
closed = cv2.dilate(closed, None, iterations=3)

closed = cv2.erode(closed, None, iterations=2)
closed = cv2.dilate(closed, None, iterations=3)
closed = cv2.dilate(closed, None, iterations=8)
# closed = cv2.dilate(closed, None, iterations=8)
closed2 = cv2.resize(closed, (960, 540))

cv2.imshow('closedBig', closed2)

cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:
    rect = cv2.boundingRect(c)
    print(rect)
    if 150 < rect[2] < 300 and 150 < rect[3] < 300:
        print(rect)
        print(cv2.contourArea(c))
        x, y, w, h = rect
        x1 = x - 100
        y1 = y - 100

        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1

        ROI = image[y1:y + h + 80, x1:x + w + 80]
        # ROI = image[y:y+h, x:x+w]

        gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

        # ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,97,43)
        # thresh1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 13)
        kernel = np.ones((5, 5), np.uint8)
        closed1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

        closed1 = cv2.erode(thresh1, None, iterations=3)
        cv2.imshow('erroded1',closed1)

        closed1 = cv2.dilate(closed1, None, iterations=1)
        cv2.imshow('closed1',closed1)
        closed1 = cv2.erode(closed1, None, iterations=2)
        cv2.imshow('erroded2',closed1)

        closed1 = cv2.dilate(closed1, None, iterations=2)
        msg1 = pylibdmtx.decode(closed1)
        print(msg1)
        cv2.imshow('ROI', ROI)
        cv2.imshow('closed', closed1)
        cv2.waitKey(0)
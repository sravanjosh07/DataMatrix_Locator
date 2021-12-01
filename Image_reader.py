import cv2
import imutils
import numpy as np
from pylibdmtx import pylibdmtx

#
# load the image and convert it to grayscale
image = cv2.imread("Img_Data/DetectTask 040021 BOTTOM 0.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# invert_img = cv2.bitwise_not(gray)
# ret, gray =cv2.threshold(invert_img, 0,255, cv2.THRESH_OTSU)
# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4

#finding horizantal lines and vertical lines
# ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F


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
blurred = cv2.blur(gradient, (5,5))
(_, thresh) = cv2.threshold(blurred, 20, 180, cv2.THRESH_BINARY)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

closed = cv2.erode(closed, None, iterations = 2)
closed = cv2.dilate(closed, None, iterations = 2)


cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for cnt in cnts:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)  # approx is a three dimensional array of vertices
    shape = np.shape(approx)
    # finding area of the contour so we can consider contours having area greater than 7000 sq.pixels
    area = cv2.contourArea(cnt)

    if area >=35000 and area <= 42250 and shape[0] == 4:
        rect = cv2.minAreaRect(cnt)
        # box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(cnt)
        # ROI = image[y :y + h, x:x + w]
        ROI = image[y - 50:y + h + 50, x - 50:x + w + 50]
        print(area)
        image1 = cv2.imread('ROI')
        gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        msg = pylibdmtx.decode(thresh1)
        print(msg)

        cv2.imshow("ROI", closed)
        cv2.waitKey(1)

###to test the pylibdmtx.decode

# image1 = cv2.imread('ROI')
# gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
# ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# msg = pylibdmtx.decode(thresh1)
# print(msg)

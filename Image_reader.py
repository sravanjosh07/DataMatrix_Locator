import cv2
import imutils
import numpy as np
from pylibdmtx import pylibdmtx

#
# load the image and convert it to grayscale
image = cv2.imread("Img_Data/DetectTask 041568 BOTTOM 0.jpg")
y = image.shape[0]
x = image.shape[1]
image[500:y-500, :] = 0
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# invert_blur = cv2.bitwise_not(gray)
# ret, gray =cv2.threshold(gray, 0,255, cv2.THRESH_OTSU)
# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4

#finding horizantal lines and vertical lines
# ddepth = cv2.cv2.CV_32F if imutils.is_cv2() else cv2.CV_32F


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
# blurred = cv2.blur(gradient, (5,5))

# ret,thresh1 = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(blurred,127,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(blurred,0,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(blurred,127,255,cv2.THRESH_TOZERO_INV)
# th2 = cv2.adaptiveThreshold(gradient,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
# ret2,th4 = cv2.threshold(gradient,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# blur = cv2.GaussianBlur(gradient,(5,5),0)
blur = cv2.bilateralFilter(gradient,9,75,75)

ret3,th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,-2)

# list1 = [thresh4, thresh5, thresh3, thresh2, thresh1,th4, th5]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 15))
closed = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

closed = cv2.erode(closed, None, iterations = 1)
closed = cv2.dilate(closed, None, iterations = 4)


closed = cv2.erode(closed, None, iterations = 2)
closed = cv2.dilate(closed, None, iterations = 4)



# cv2.imshow("closed1", closed)
# cv2.waitKey(0)
#
cnts = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:
    rect = cv2.boundingRect(c)
    if rect[2] < 150 or rect[3] < 150: continue
    print(rect)

    print(cv2.contourArea(c))
    x,y,w,h = rect
    # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.putText(image,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))

    ROI = image[y:y + h, x:x + w]
    print(ROI)
    # ROI = image[y - 50:y + h + 50, x - 50:x + w + 50]

    # image1 = cv2.imread('ROI')
    gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("thresh", thresh1)

    closed1 = cv2.erode(thresh1, None, iterations=2)
    closed1 = cv2.dilate(closed1, None, iterations=2)
    cv2.imshow("closed1", closed1)

    closed1 = cv2.erode(closed1, None, iterations=3)
    closed1 = cv2.dilate(closed1, None, iterations=3)
    cv2.imshow("closed2", closed1)

    closed1 = cv2.erode(closed1, None, iterations=4)
    closed1 = cv2.dilate(closed1, None, iterations=4)
    cv2.imshow("closed3", closed1)

    closed1 = cv2.erode(closed1, None, iterations=5)
    closed1 = cv2.dilate(closed1, None, iterations=5)
    cv2.imshow("closed4", closed1)

    closed1 = cv2.erode(closed1, None, iterations=1)
    closed1 = cv2.dilate(closed1, None, iterations=1)
    cv2.imshow("closed5", closed1)


    cv2.imwrite("ROI.jpg", ROI, params= None)

    msg = pylibdmtx.decode(gray1)
    print(msg)
    # cv2.imshow("closed1", closed)
    # cv2.imshow("closed3", th3)
    # cv2.imshow("blur", closed1)
    cv2.imshow("thresh", ROI)
    cv2.waitKey(0)
    # cv2.destroyWindow('closed1')

# cv2.drawContours(image, cnts, -1, (0,255,0), 3)
# cv2.imshow("closed1", image)
# cv2.waitKey(0)

#
# for cnt in cnts:
#     peri = cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)  # approx is a three dimensional array of vertices
#     shape = np.shape(approx)
#     # finding area of the contour so we can consider contours having area greater than 7000 sq.pixels
#     area = cv2.contourArea(cnt)
#
#     if area >= 20000 and area <= 45550 and shape[0] == 4:
#         rect = cv2.minAreaRect(cnt)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#         cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
#         continue
#     continue



#
#     if area >=35000 and area <= 42250 and shape[0] == 4:
#         rect = cv2.minAreaRect(cnt)
#         # box = cv2.cv2.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
#         # box = np.int0(box)
#         # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
#         x, y, w, h = cv2.boundingRect(cnt)
#         # ROI = image[y :y + h, x:x + w]
#         ROI = image[y - 50:y + h + 50, x - 50:x + w + 50]
#         print(area)
#         # image1 = cv2.imread('ROI')
#         gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
#         ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         msg = pylibdmtx.decode(thresh1)
#         print(msg)
#
#         cv2.imshow("ROI", ROI)
#             cv2.waitKey(0)


#     cv2.imshow("closed", closed)
#     cv2.waitKey(0)
# ###to test the pylibdmtx.decode

# image1 = cv2.imread('ROI')
# gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
# ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# msg = pylibdmtx.decode(thresh1)
# print(msg)

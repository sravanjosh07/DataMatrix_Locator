#
#
#
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
#
#
# class ContourLocator:
#     """ Utility for finding the positions of all of the datamatrix barcodes
#     in an image """
#
#     def __init__(self):
#         pass
#
#     def locate_datamatrices(self, image):
#         """Get the positions of (hopefully all) datamatrices within an image.
#         """
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
#         # Perform adaptive threshold, reducing to a binary image
#
#         threshold_image = self._do_threshold(gray_image, 3, 10)
#
#         # Perform a morphological close, removing noise and closing some gaps
#         morphed_image = self._do_close_morph(threshold_image, 10)
#
#         # Find a bunch of contours in the image.
#         contours = self._get_contours(morphed_image)
#         polygons = self._contours_to_polygons(contours)
#         return polygons
#     def _do_threshold(self, gray_image, block_size, c):
#         """ Perform an adaptive threshold operation on the image. """
#         raw = gray_image
#         thresh = cv2.adaptiveThreshold(raw, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
#         return thresh
#
#     #@staticmethod
#     def _do_close_morph(self, threshold_image, morph_size):
#         """ Perform a generic morphological operation on an image. """
#         element = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
#         closed = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, element, iterations=1)
#         return closed
#
#     #@staticmethod
#     def _get_contours(self, binary_image):
#         """ Find contours and return them as lists of vertices. """
#         raw_img = binary_image.copy()
#
#         # List of return values changed between version 2 and 3
#         raw_contours, _ = cv2.findContours(raw_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#         return raw_contours
#
#     # @staticmethod
#     def _contours_to_polygons(self, contours, epsilon=6.0):
#         """ Uses the Douglas-Peucker algorithm to approximate a polygon as a similar polygon with
#         fewer vertices, i.e., it smooths the edges of the shape out. Epsilon is the maximum distance
#         from the contour to the approximated contour; it controls how much smoothing is applied.
#         A lower epsilon will mean less smoothing. """
#         shapes = [cv2.approxPolyDP(rc, epsilon, True).reshape(-1, 2) for rc in contours]
#         return shapes
#
#
# a = ContourLocator()
# b= cv2.imread("test1.jpg")
# a.locate_datamatrices(b)

#{{Second Method}}

# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
#
# well = plt.imread('test1.jpg')
# well = cv2.cvtColor(well, cv2.COLOR_BGRA2GRAY)
# plt.subplot(151); plt.title('A')
# plt.imshow(well)
#
# harris = cv2.cornerHarris(well,4, 1,0.00)
# plt.subplot(152); plt.title('B')
# plt.imshow(harris)
#
# x, thr = cv2.threshold(harris, 0.1 * harris.max(), 255, cv2.THRESH_BINARY)
# thr = thr.astype('uint8')
# plt.subplot(153); plt.title('C')
# plt.imshow(thr)
#
# contours, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# areas = map(lambda x: cv2.contourArea(cv2.convexHull(x)), contours)
# areas = list(areas)
# max_i = areas.index(max(areas))
# d = cv2.drawContours(np.zeros_like(thr), contours, max_i, 255, 1)
# plt.subplot(154); plt.title('D')
# plt.imshow(d)
#
# rect =cv2.minAreaRect(contours[max_i])
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# e= cv2.drawContours(well,[box],0,1,1)
# plt.subplot(155); plt.title('E')
# plt.imshow(e)
#
# plt.show()


# import  cv2
# import imutils
# import numpy as np
# # load the image and convert it to grayscale
# image = cv2.imread("test2.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # compute the Scharr gradient magnitude representation of the images
# # in both the x and y direction using OpenCV 2.4
# ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
#
# kernelx = np.array([[-1, 0, +1],
#                    [-2, 0, +2],
#                    [-1, 0, +1]])
# gradX = cv2.filter2D(gray, -1, kernelx)
#
# kernely = np.array([[-1, -2, -1],
#                    [0, 0, 0],
#                    [+1, +2, -1]])
# gradY = cv2.filter2D(gray, -1, kernely)
#
# # subtract the y-gradient from the x-gradient
# gradient = cv2.subtract(gradX, gradY)
# gradient = cv2.convertScaleAbs(gradient)
#
# blurred = cv2.blur(gradient, (5,5))
# (_, thresh) = cv2.threshold(blurred, 20, 180, cv2.THRESH_BINARY)
#
#
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 25))
# closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#
# closed = cv2.erode(closed, None, iterations = 2)
# closed = cv2.dilate(closed, None, iterations = 2)
#
#
# cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# for cnt in cnts:
#     peri = cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)  # approx is a three dimensional array of vertices
#     shape = np.shape(approx)
#     # finding area of the contour so we can consider contours having area greater than 7000 sq.pixels
#     area = cv2.contourArea(cnt)
#     print(area)
#     if area >=30000 and area <= 45000 and shape[0] == 4:
#         print("hey, Im running")
#         rect = cv2.minAreaRect(cnt)
#         box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
#         box = np.int0(box)
#         cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
#         cv2.imshow("Image", image)
#         cv2.waitKey(0)
        # cv2.drawContours(image, i, -1, (255, 255, 255), 3)
        # rect = cv2.minAreaRect(c)
        # box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
        # box = np.int0(box)
# draw a bounding box arounded the detected barcode and display the
# image
# rect = cv2.minAreaRect(c)
# box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
# box = np.int0(box)
# # draw a bounding box arounded the detected barcode and display the
# # image
# cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# # #
# cv2.imshow("gradient", closed)
# cv2.waitKey()
#
#
import  cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from pylibdmtx import pylibdmtx
from pylibdmtx.pylibdmtx import decode
#
# #
# # load the image and convert it to grayscale
image = cv2.imread("Img_Data/test1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4

#finding horizantal lines and vertical lines
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F

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

    if area >=30000 and area <= 45000 and shape[0] == 4:
        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
        box = np.int0(box)
        x, y, w, h = cv2.boundingRect(cnt)
        #ROI = image[y :y + h, x:x + w]
        ROI = image[y- 20 :y + h+ 20, x-20:x + w+20]
        print(area)
        image1 = cv2.imread('ROI')
        gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


        msg = pylibdmtx.decode(thresh1)
        print(msg)
        # # data = decode(thresh1)
        cv2.imshow("Thresh", thresh1)
        cv2.waitKey(0)

#         # print(data)
#
#         # cv2.namedWindow("Largest Contour", cv2.WINDOW_NORMAL)
#         # image1 = cv2.imread('ROI')
#         #
#         # gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
#         #
#         # ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         #
#         # msg = pylibdmtx.decode(thresh1)
#         # print(msg)
#
#         #cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
#         print(area)



# image1 = cv2.imread('Img_Data/customer_vision-comp_dm_dot.jpg')
# gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
# cv2.imshow("thresh", thresh1)
# cv2.waitKey(0)
# msg = pylibdmtx.decode(thresh1)
# print(msg)

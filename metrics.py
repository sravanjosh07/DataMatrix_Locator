import cv2
import imutils
import numpy as np
from pylibdmtx.pylibdmtx import decode
import os

mycode_list = []
path = 'Img_Data'
output_path = 'metrics.csv'
border_width = 500


def preprocessing(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernelx = np.array([[-1, 0, +1],
                        [-2, 0, +2],
                        [-1, 0, +1]])
    gradX = cv2.filter2D(gray_image, -1, kernelx)

    kernely = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [+1, +2, -1]])
    gradY = cv2.filter2D(gray_image, -1, kernely)

    # subtract the y-gradient from the x-gradient to learn
    gradient = cv2.subtract(gradX, gradY)
    # gradient = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
    gradient = cv2.convertScaleAbs(gradient)

    return gradient


def blur_and_morph(gradient_image):
    blur = cv2.bilateralFilter(gradient_image, 9, 75, 75)

    threshold_image = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, -2)

    # list1 = [thresh4, thresh5, thresh3, thresh2, thresh1,th4, th5]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 15))
    closed_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)

    closed_image = cv2.erode(closed_image, None, iterations=2)
    closed_image = cv2.dilate(closed_image, None, iterations=2)

    closed_image = cv2.erode(closed_image, None, iterations=2)
    closed_image = cv2.dilate(closed_image, None, iterations=3)

    closed_image = cv2.erode(closed_image, None, iterations=2)
    closed_image = cv2.dilate(closed_image, None, iterations=3)
    closed_image = cv2.dilate(closed_image, None, iterations=2)

    return closed_image


def grab_contours(closed_image):
    contours = cv2.findContours(closed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return contours


def find_roi(rectangle, frame):
    x, y, w, h = rectangle
    x1 = x - 75
    y1 = y - 75

    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1

    roi = frame[y1:y + h + 75, x1:x + w + 75]
    return roi


def closing_on_roi(region_of_interest):
    gray1 = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
    # ret, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 97, 43)
    closed1 = cv2.erode(thresh1, None, iterations=3)
    closed1 = cv2.dilate(closed1, None, iterations=1)
    closed1 = cv2.erode(closed1, None, iterations=2)
    closed1 = cv2.dilate(closed1, None, iterations=2)
    return closed1


# def decode_contour(rect):
#     x, y, w, h = rect
#     x1 = x - 75
#     y1 = y - 75
#
#     x1 = 0 if x1 < 0 else x1
#     y1 = 0 if y1 < 0 else y1
#
#     ROI = frame[y1:y + h + 75, x1:x + w + 75]
#
#     gray1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
#     thresh1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 97, 43)
#     kernel = np.ones((5, 5), np.uint8)
#     closed1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
#     closed1 = cv2.erode(closed1, None, iterations=3)
#     closed1 = cv2.dilate(closed1, None, iterations=1)
#     closed1 = cv2.erode(closed1, None, iterations=2)
#     closed1 = cv2.dilate(closed1, None, iterations=2)
#
#     message = decode(closed1)
#     return message


path = "img_data/DetectTask 049265 BOTTOM 0.jpg"
output_path = "metrics.csv"
dictionary_of_Boolean = {}
image = cv2.imread(path)
gradient_image = preprocessing(image)
closed_image = blur_and_morph(gradient_image)
contours = cv2.findContours(closed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
for contour in contours:
    rect = cv2.boundingRect(contour)
    if 150 < rect[2] < 300 and 150 < rect[3] < 300:
        print(rect)
        roi = find_roi(rect, image)
        closed = closing_on_roi(roi)
        code = decode(closed)
        print(code)
        cv2.imshow('roi', roi)
        cv2.imshow('closed', closed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# for filename in os.listdir(path):
#     if filename.endswith('.jpg'):
#         dictionary_of_Boolean[filename] = 0
#         input_path = os.path.join(path, filename)
#
#         frame = cv2.imread(input_path)
#         gradient_image = preprocessing(frame)
#         closed_image = blur_and_morph(gradient_image)
#         contours = grab_contours(closed_image)
#         for contour in contours:
#             rect = cv2.boundingRect(contour)
#             if 150 < rect[2] < 300 and 150 < rect[3] < 300:
#                 print(rect)
#                 code_data = decode_contour(rect)
#                 print(code_data)
#                 if code_data:
#                     dictionary_of_Boolean[filename] = 1
#                     break
#     with open('metrics3.csv', "a") as csv:
#
#         csv.write("{},{}\n".format(filename, dictionary_of_Boolean[filename]))
#         csv.flush()

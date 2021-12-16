class Detect:
    def __init__(self, frame):
        self.frame = frame
        self.image_for_roi = frame

    @classmethod
    def locate_data_matrix(cls, frame):
        gradient_image = Detect.gradient_operation(frame)
        closed_image = Detect.blur_and_morph(gradient_image)
        rois = Detect.find_required_contours(closed_image)
        closed_rois = Detect.preprocessing_roi(rois)
        difficult_to_read_roi = Detect.preprocessing_difficult_roi(rois)
        my_code = Detect.extract_code(closed_rois)
        my_difficult_code = Detect.extract_code(difficult_to_read_roi)
        my_code = my_difficult_code if not my_code else my_code
        return my_code

    @staticmethod
    def gradient_operation(img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernelx = np.array([[-1, 0, +1],
                            [-2, 0, +2],
                            [-1, 0, +1]])
        gradX = cv2.filter2D(gray, -1, kernelx)

        kernely = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [+1, +2, -1]])
        gradY = cv2.filter2D(gray, -1, kernely)

        """ 
        # subtract the y-gradient from the x-gradient to extract contours that has a horizontal and vertical line in them.
        an "L" shape is a characteristic property of a Data Matrix. 
        """
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        return gradient

    @staticmethod
    def blur_and_morph(gradient):
        # blurred = cv2.GaussianBlur(gradient, (7, 11), 0)
        blurred = cv2.bilateralFilter(gradient, 9, 75, 75)
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        """
        Morphing with a 40x25 kernel to fill the gaps in the detected contour"
        Dilating the contour more than eroding, because the contour contains white pixels.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 15))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        closed = cv2.erode(morphed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=3)

        return closed

    @staticmethod
    def find_required_contours(closed):
        rois = []
        contour_offset = 30
        side_min = 150
        contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = imutils.grab_contours(contours)
        for contour in contours:
            rect = cv2.boundingRect(contour)
            if side_min < rect[2] < 2 * side_min and side_min < rect[3] < 2 * side_min:
                print(rect)
                x, y, w, h = rect

                x1 = x - contour_offset
                y1 = y - contour_offset
                h1 = h + contour_offset
                w1 = w + contour_offset
                x1 = 0 if x1 < 0 else x1
                y1 = 0 if y1 < 0 else y1
                h1 = 0 if h1 < 0 else h1
                w1 = 0 if w1 < 0 else w1
                roi = image_for_roi[y1:y + h1, x1:x + w1]
                rois.append(roi)
        return rois

    @staticmethod
    def preprocessing_roi(rois):
        closed_rois = []
        for roi in rois:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            closed1 = cv2.erode(thresh, None, iterations=2)
            closed1 = cv2.dilate(closed1, None, iterations=2)
            closed_rois.append(closed1)
        return closed_rois

    @staticmethod
    def preprocessing_difficult_roi(rois):
        difficult_rois = []
        for roi in rois:
            gray1 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,12)
            closed1 = cv2.erode(thresh1, None, iterations=3)
            closed1 = cv2.dilate(closed1, None, iterations=1)

            closed1 = cv2.erode(closed1, None, iterations=2)
            closed1 = cv2.dilate(closed1, None, iterations=2)
            difficult_rois.append(closed1)
        return difficult_rois

    @staticmethod
    def extract_code(closed_rois):
        for closed_roi in closed_rois:
            for matrix_code in decode(closed_roi):
                if matrix_code:
                    code_data = matrix_code.data.decode('utf-8')
                    return code_data
                else:
                    return None


import cv2
import numpy as np
import os
from pylibdmtx.pylibdmtx import decode
import imutils

path = 'img_data'
# input_path = 'img_data/DetectTask 054300 BOTTOM 0.jpg'
# frame = cv2.imread(input_path)
# image_for_roi = frame
# code = Detect.locate_data_matrix(frame)
# print(code)
for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(path, filename)
        frame = cv2.imread(input_path)
        image_for_roi = frame
        code = Detect.locate_data_matrix(frame)
        with open('metrics3.csv', "a") as csv:
            if code:
                csv.write("{},{}\n".format(filename, 1))
                print(code)
                continue
            else:
                csv.write("{},{}\n".format(filename, 0))
                csv.flush()
                continue
        print(code)

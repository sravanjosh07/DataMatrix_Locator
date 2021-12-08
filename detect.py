class Detect:
    def __init__(self, frame):
        self.frame = frame
        self.frame_for_roi = frame


    @classmethod
    def locate_data_matrix(cls, frame):
        preprocessed_image = Detect.preprocessed(frame)
        gradient_image = Detect.gradient_operation(preprocessed_image)
        closed_image = Detect.blur_and_morph(gradient_image)
        roi = Detect.find_required_contours(closed_image)
        thresh_roi = Detect.preprocessing_roi(roi)
        erroded_roi = Detect.apply_additional_erosion(thresh_roi)
        try:
            code = Detect.decode_roi(thresh_roi)
        except:
            code = Detect.decode_roi(erroded_roi)

        return code

    @staticmethod
    def gradient_operation(gray):
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
        blurred = cv2.GaussianBlur(gradient, (7, 11), 0)
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        """
        Morphing with a 40x25 kernel to fill the gaps in the detected contour"
        Dilating the contour more than eroding, because the contour contains white pixels.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 25))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        closed = cv2.erode(morphed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=3)

        return closed

    @staticmethod
    def preprocessed(img):
        assert img is not None
        y = img.shape[0]
        x = img.shape[1]
        img[500:y - 500, :] = 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def find_required_contours(closed):
        contour_offset = 50
        contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)  # approx is a three dimensional array of vertices
            shape = np.shape(approx)
            # finding area of the contour so we can consider contours having area greater than 7000 sq.pixels
            area = cv2.contourArea(contour)
            if 25000 <= area <= 45550 and shape[0] == 4:
                rect = cv2.minAreaRect(contour)
                x, y, w, h = cv2.boundingRect(contour)

                x1 = x - contour_offset
                y1 = y - contour_offset
                h1 = h + contour_offset
                w1 = w + contour_offset
                x1 = 0 if x1 < 0 else x1
                y1 = 0 if y1 < 0 else y1
                roi = image_for_roi[y1:y + h1, x1:x + w1]
                return roi
            else:
                continue

    @staticmethod
    def preprocessing_roi(roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        closed1 = cv2.erode(thresh, None, iterations=2)
        closed1 = cv2.dilate(closed1, None, iterations=2)

        return closed1

    @staticmethod
    def extract_code(closed1):
        try:
            Detect.is_code(code)
        except:
            Detect.apply_additional_erosion(closed1)
            De





    @staticmethod
    def decode_roi(closed1):
        for matrix_code in decode(closed1):
            if matrix_code is not None:
                return matrix_code
            else:
                return False



    # @staticmethod
    # def is_code(code):
    #     if code is not None:
    #         return True

    @staticmethod
    def apply_additional_erosion(closed1):

        closed1 = cv2.erode(closed1, None, iterations=3)
        closed1 = cv2.dilate(closed1, None, iterations=3)

        closed1 = cv2.erode(closed1, None, iterations=4)
        closed1 = cv2.dilate(closed1, None, iterations=4)

        closed1 = cv2.erode(closed1, None, iterations=5)
        closed1 = cv2.dilate(closed1, None, iterations=5)

        return closed1

    # @staticmethod
    # def write_to_csv(matrix_code):
    #     with open('metrics.csv', "a") as csv:
    #         csv.write("{},{}\n".format(filename, 1))
    #



import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode

# path = 'Img_Data'
path = '/Users/Sravan/Desktop/DataMatrix_Locator-master-2/Img_Data/DetectTask 041576 BOTTOM 0.jpg'
frame = cv2.imread(path)
image_for_roi = frame
code = Detect.locate_data_matrix(frame)
print(code)




# for filename in os.listdir(path):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         input_path = os.path.join(path,filename)
#         image_for_roi = cv2.imread(input_path)
#         masked_img = np.zeros(image_for_roi.shape, dtype= np.uint8)
#         frame = cv2.imread(input_path)
#         preprocessed = Detect.preprocessed(frame)
#         gradient = Detect.gradient_operation(preprocessed)
#         closed1 = Detect.blur_and_morph(gradient)
#         roi = Detect.find_required_contours(closed1)
#         thresh_roi = Detect.preprocessing_roi(roi) if roi is not None else masked_img
#         code = Detect.decode_roi(thresh_roi)
#         with open('metrics.csv', "a") as csv:
#             if code is not None:
#                 csv.write("{},{}\n".format(filename, 1))
#                 continue
#                 print(code)
#             else:
#                 csv.write("{},{}\n".format(filename, 0))
#                 csv.flush()
#                 continue
#             print(code)
#     # cv2.imshow("roi", roi)
#     # cv2.imshow('closed', closed1)
#     # cv2.waitKey(0)
#
#

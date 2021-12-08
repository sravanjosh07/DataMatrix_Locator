import cv2
import imutils
import numpy as np
from pylibdmtx import pylibdmtx
import datetime

camera = True
cap = cv2.VideoCapture(0)
scale_percent = 80
scanned_code_list = []
scanned_data_dict : dict[str, datetime.datetime] = {}  # key: code, value: last detection


def gradient_operation(frame):
    kernelx = np.array([[-1, 0, +1],
                        [-2, 0, +2],
                        [-1, 0, +1]])
    gradX = cv2.filter2D(frame, -1, kernelx)

    kernely = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [+1, +2, -1]])
    gradY = cv2.filter2D(frame, -1, kernely)

    # subtract the y-gradient from the x-gradient to learn
    gradient = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
    gradient = cv2.convertScaleAbs(gradient)

    return gradient

def blur_and_morph(image):
    blurred = cv2.blur(image, (5,5))
    (_, thresh1) = cv2.threshold(blurred, 20, 180, cv2.THRESH_BINARY)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 25))
    closed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations = 2)
    closed = cv2.dilate(closed, None, iterations = 2)
    return closed


def preprocessed(frame):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame


while True:
    ret, frame = cap.read()
    frame = preprocessed(frame)
    gradient = gradient_operation(frame)
    closed = blur_and_morph(gradient)

    # Find Contours
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Loop through contour to find the contours with required area
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)  # approx is a three dimensional array of vertices
        shape = np.shape(approx)
        # finding area of the contour so we can consider contours having area greater than 7000 sq.pixels
        area = cv2.contourArea(cnt)

        if area >=30000 and area <= 42250: #and shape[0] == 4:
            captured = "save_image.jpg"
            cv2.imwrite(captured, frame)
            rect = cv2.minAreaRect(cnt)

            x, y, w, h = cv2.boundingRect(cnt)
            ROI = frame[y - 100:y + h + 100, x - 100:x + w + 100]
            # ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            ret, ROI = cv2.threshold(ROI, 0, 255, cv2.THRESH_OTSU)
            if ROI is not None:
                for code in pylibdmtx.decode(ROI):
                    mycode = code.data.decode("utf-8")

                    if mycode not in scanned_code_list:
                        print(mycode)
                        scanned_code_list.append(mycode)
                        scanned_data_dict["mycode"] = x

                    elif mycode in scanned_code_list and x - scanned_data_dict["mycode"] > 5:
                        # scanned_data_dict.update({"mycode": x}) #clearing the list will rewrite the dict as it's going to get scanned again
                        scanned_code_list.clear()
                        continue

                    else:
                        continue

                cv2.imshow('ROI', ROI)
    cv2.imshow('thresh', frame)
    cv2.waitKey(1)


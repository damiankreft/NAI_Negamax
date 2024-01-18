""" Authors:Damian Kreft, Sebastian Kreft
    Required environment: Python3, opencv-python """

from time import sleep
import numpy as np
import math
import cv2 as cv
import subprocess
import datetime


def GestureControls():
    """Program allowes user to use hand gestures to perform four predefined actions. 
    Showing two finger opens youtube page
    Showing three finger opens Gakko page
    Showing four finger opens notepad
    Showing five finger puts computer to sleep"""

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))

    time = datetime.datetime.now()
    lastActionId = 1

    while (True):
        _, img = cap.read()
        cv.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
        crop_img = img[100:300, 100:300]
        grey = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        value = (35, 35)
        blurred_ = cv.GaussianBlur(grey, value, 0)
        # _, thresholded = cv.threshold(blurred_, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        _, threshold = cv.threshold(blurred_, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        contours, hierarchy = cv.findContours(threshold.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours1 = max(contours, key=lambda x: cv.contourArea(x))
        x, y, w, h = cv.boundingRect(contours1)
        cv.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)
        hull = cv.convexHull(contours1)
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv.drawContours(drawing, [contours1], 0, (0, 255, 0), 0)
        cv.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
        hull = cv.convexHull(contours1, returnPoints=False)
        defects = cv.convexityDefects(contours1, hull)

        count_defects = 0
        cv.drawContours(threshold, contours, -1, (0, 255, 0), 3)

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contours1[s][0])
            end = tuple(contours1[e][0])
            far = tuple(contours1[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            if angle <= 90:
                count_defects += 1
                cv.circle(crop_img, far, 1, [0, 0, 255], -1)

            cv.line(crop_img, start, end, [0, 255, 0], 2)

        text = str(count_defects) + " finger(s)"
        if (datetime.datetime.now() - time).seconds > 2 and count_defects != 1:
            time = datetime.datetime.now()
            
            if count_defects == 1:
                break
            elif count_defects == 2 and lastActionId == 2:
                print("2 fingers")
                subprocess.run(["C:\\Program Files\\Mozilla Firefox\\firefox.exe", "https://www.youtube.com/watch?v=jC3pGcRHeQ0"])
            elif count_defects == 3 and lastActionId == 3:
                print("3 fingers")
                subprocess.run(["C:\\Program Files\\Mozilla Firefox\\firefox.exe", "https://gakko.pjwstk.edu.pl"])
            elif count_defects == 4 and lastActionId == 4:
                print("4 fingers")
                subprocess.run(["notepad", str(datetime.datetime.now())])
            elif count_defects == 5 and lastActionId == 5:
                print("5 fingers")
                subprocess.run(["shutdown", "/i"])

            lastActionId = count_defects

        cv.putText(img, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)
        cv.imshow('Captured picture', img)
        cv.imshow('Contour helper window', drawing)
        if cv.waitKey(10) == 27:
            break

GestureControls()
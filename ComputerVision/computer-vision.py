import numpy as np
import math
import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))

while (True):
    _, img = cap.read()
    cv.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
    crop_img = img[100:300, 100:300]
    grey = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
    value = (35, 35)
    blurred_ = cv.GaussianBlur(grey, value, 0)
    # _, thresholded = cv.threshold(blurred_, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    _, thresholded = cv.threshold(blurred_, 64, 255, cv.THRESH_OTSU)

    contours, hierarchy = cv.findContours(thresholded.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    count1 = max(contours, key=lambda x: cv.contourArea(x))
    x, y, w, h = cv.boundingRect(count1)
    cv.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)
    hull = cv.convexHull(count1)
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv.drawContours(drawing, [count1], 0, (0, 255, 0), 0)
    cv.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
    hull = cv.convexHull(count1, returnPoints=False)
    defects = cv.convexityDefects(count1, hull)

    count_defects = 0
    cv.drawContours(thresholded, contours, -1, (0, 255, 0), 3)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(count1[s][0])
        end = tuple(count1[e][0])
        far = tuple(count1[f][0])
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

        if angle <= 90:
            count_defects += 1
            cv.circle(crop_img, far, 1, [0, 0, 255], -1)

        cv.line(crop_img, start, end, [0, 255, 0], 2)

    text = str(count_defects + 1) + " finger(s)"
    cv.putText(img, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)
    cv.imshow('Captured picture', img)
    cv.imshow('Contour helper window', drawing)
    if cv.waitKey(10) == 27:
        break
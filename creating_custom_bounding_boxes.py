import cv2
import numpy

cap = cv2.VideoCapture(0)









while True:
    ## image capture
    success,img = cap.read()


    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(grey, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    ## Image read and output waitkey


    cv2.imshow("YEA",img)
    cv2.waitKey(1)

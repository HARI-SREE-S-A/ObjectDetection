import cv2
import numpy as np
import time

##open cv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale = 1/255)
## capture the image
cap = cv2.VideoCapture(0)

while True:
    ## getting the frame
    success,frame = cap.read()
    ## object detection

    (class_ids,scores,bboxs) = model.detect(frame)
    for class_id,score,bbox in zip(class_ids,scores,bboxs):
        x,y,w,h = bbox
        print(x,y,w,h)
    #print(class_ids,scores,bboxs)





    cv2.imshow("harvis",frame)
    cv2.waitKey(1)





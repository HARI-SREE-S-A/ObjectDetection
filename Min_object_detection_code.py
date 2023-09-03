import cv2
import numpy as np
import time

##open cv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale = 1/255)

#### classes list
classes = []
with open("dnn_model/classes.txt","r") as file:
    for class_name in file:
        class_name = class_name.strip()
        classes.append(class_name)
#print(classes)
## capture the image
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

while True:
    ## getting the frame
    success,frame = cap.read()
    // object detection

    (class_ids,scores,bboxs) = model.detect(frame)
    for class_id,score,bbox in zip(class_ids,scores,bboxs):
        x,y,w,h = bbox
        classname = classes[class_id]
        print(x,y,w,h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,50),4)
        cv2.putText(frame,str(classname),(x,y-10),cv2.FONT_HERSHEY_PLAIN,2,(200,0,50),2)
        #print(class_ids,scores,bboxs)






    cv2.imshow("harvis",frame)
    cv2.waitKey(1)





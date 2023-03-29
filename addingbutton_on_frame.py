import cv2
import numpy as np
import time


button_person = False


##open cv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

#### classes list
classes = []
with open("dnn_model/classes.txt", "r") as file:
    for class_name in file:
        class_name = class_name.strip()
        classes.append(class_name)
# print(classes)
## capture the image
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
        cv2.fillPoly(frame, polygon, (0, 0, 200))
        #print(x, y)
        ##check if inside
        #is_inside = cv2.pointPolygonTest(polygon,(x,y),False)
        #if is_inside >0:
            #print("clicked")
            #if button_person is False:
                #button_person = True
            #else:
                #button_person = False
            #print("now button pressed ",button_person)



cv2.namedWindow("harvis")
cv2.setMouseCallback("harvis", click_button)

while True:
    ## getting the frame
    success, frame = cap.read()
    ## object detection

    (class_ids, scores, bboxs) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxs):
        x, y, w, h = bbox
        classname = classes[class_id]
        # print(x,y,w,h)
        if classname == "person" and button_person == True:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 4)
            cv2.putText(frame, str(classname), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
            # print(class_ids,scores,bboxs)

    #cv2.rectangle(frame,(20,20),(150,70),(0,100,100),-1)

    #cv2.fillPoly(frame, polygon, (0, 0, 200))
    #cv2.putText(frame, str("person"), (30, 60), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 3)

    cv2.imshow("harvis", frame)
    cv2.waitKey(1)


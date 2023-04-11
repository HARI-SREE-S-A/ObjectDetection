from ultralytics import YOLO


model = YOLO("yolov8m_custom.pt")

model.train(data="data_custom.yaml",batch=8 , imgsz=640, epochs=100, workers=1)
def pred():
    model.predict(source=0,show=True,save=True,con=0.5)

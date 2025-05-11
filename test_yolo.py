from ultralytics import YOLO
import cv2

model=YOLO("yolov8n.pt")
#results = model(source=..., stream=True)
model.track(source=0,conf=0.3,iou=0.5,show=True)
# model.predict(source=0,show=True)



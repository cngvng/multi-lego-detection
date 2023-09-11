import cv2
from ultralytics import YOLO

model = YOLO(model="runs/train5/weights/best.pt")

results = model("/Users/cngvng/Desktop/multi-lego-detection/test/IMG_7671.jpeg")

bounding_boxes = results[0].boxes

confidence_threshold = 0.8
if bounding_boxes is not None and len(bounding_boxes) > 0:
    for i, box in enumerate(bounding_boxes):
        confidence = box.conf.item()
        # if confidence > confidence_threshold:
        #     cv2.imshow("YOLO detection", box)
        print(box.cls)
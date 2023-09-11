import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', classes=6)
model.load_state_dict(torch.load('runs/train/yolo_lego_det/weights/best.pt')['model'].state_dict())

image_path = "/Users/cngvng/Desktop/multi-lego-detection/test/IMG_7671.jpeg"
frame = cv2.imread(image_path)

# Chuyển hình ảnh thành tensor và chuẩn bị cho việc truyền vào model
img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0)  # Thêm một chiều batch size

# Truyền tensor hình ảnh vào model
results = model(img_tensor)
# results = model("test/IMG_7671.jpeg")
import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
yolo = YOLO(model="/Users/cngvng/Desktop/multi-lego-detection/runs/train5/weights/best.pt")

st.title("YOLOv8 Object Detection Demo")
st.sidebar.header("Settings")
# confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Open a video stream from the camera (0 is typically the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Unable to open camera.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to read frame.")
            break

        # Perform object detection
        # detected_frame = yolo.detect(frame, confidence_threshold)
        results = yolo(frame)

        annotationed = results[0].plot()

        # Convert the frame to RGB (required by PIL for display)
        # rgb_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)

        # Display the frame with detections
        st.image(annotationed, channels="BGR")

# Release the camera when done
cap.release()

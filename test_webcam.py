#!/usr/bin/env python3
"""
Quick test script for YOLOv8 using webcam (for testing before RTSP)
"""
from ultralytics import YOLO

# Initialize YOLOv8 model (will download if not present)
model = YOLO('yolov8n.pt')  # yolov8n = nano (fastest), yolov8s/m/l/x for better accuracy

# Run inference on webcam (source=0)
# Press 'q' to quit
results = model.predict(source=0, show=True, conf=0.25)

print("Webcam test complete!")

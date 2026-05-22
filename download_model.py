"""
SignSense AI - Download Pre-trained Model
Downloads YOLOv5 small model and saves it to model/best.pt
"""

from ultralytics import YOLO
import os

print("🧠 Downloading YOLOv5 pre-trained model...")
print("This may take a few minutes...\n")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Download and save yolov5s model
model = YOLO('yolov5s.pt')
model.save('model/best.pt')

print("\n✓ Model downloaded successfully!")
print("✓ Saved to: model/best.pt")
print("\nYou can now run: python detect.py")

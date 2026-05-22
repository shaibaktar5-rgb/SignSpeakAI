"""
SignSense AI - Configuration File
Adjust these settings to customize detection behavior
"""

# Camera Settings
CAMERA_ID = 0  # 0 = default camera, 1 = second camera, etc.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Detection Settings
CONFIDENCE_THRESHOLD = 0.5  # 0-1: Lower = more detections, Higher = more confident
IOU_THRESHOLD = 0.45  # Intersection over Union for NMS (Non-Maximum Suppression)

# Model Settings
MODEL_PATH = "model/best.pt"  # Path to your trained YOLO model
USE_GPU = True  # Set to False if you don't have CUDA

# Display Settings
SHOW_FPS = True  # Show frames per second
SHOW_CONFIDENCE = True  # Show confidence scores
LINE_THICKNESS = 2  # Bounding box line thickness
FONT_SCALE = 0.6  # Text size

# Output Settings
SAVE_DETECTIONS = False  # Save detected frames
OUTPUT_FOLDER = "detections/"  # Where to save frames

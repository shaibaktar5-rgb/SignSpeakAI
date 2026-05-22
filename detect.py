"""
SignSense AI - Real-time Sign Language Detection
Uses YOLOv5 to detect hand signs from webcam or images
"""

import cv2
import os
from pathlib import Path
from ultralytics import YOLO

# Load YOLOv5 model
def load_model(model_path="model/best.pt"):
    """
    Load the trained YOLOv5 model
    
    Args:
        model_path: Path to the trained model file
    
    Returns:
        Loaded model object
    """
    print(f"Loading model from {model_path}...")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please place your trained best.pt file in the model/ folder")
        return None
    
    try:
        # Load YOLOv5 model using Ultralytics
        model = YOLO(model_path)
        print("✓ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None


def detect_from_webcam(model, confidence=0.5):
    """
    Run real-time detection from webcam
    
    Args:
        model: Loaded YOLOv5 model
        confidence: Confidence threshold (0-1)
    """
    if model is None:
        return
    
    print("Starting webcam...")
    print("Press 'q' to quit\n")
    
    # Open webcam (0 is default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("ERROR: Cannot read frame from webcam")
                break
            
            # Run inference on the frame
            results = model(frame, conf=confidence)
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Display the frame
            cv2.imshow("SignSense AI - Real-time Detection", annotated_frame)
            
            # Print detections
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = results[0].names[cls]
                    print(f"Detected: {label} (Confidence: {conf:.2f})")
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def detect_from_image(model, image_path, confidence=0.5):
    """
    Run detection on a single image
    
    Args:
        model: Loaded YOLOv5 model
        image_path: Path to the image file
        confidence: Confidence threshold (0-1)
    """
    if model is None:
        return
    
    print(f"Loading image from {image_path}...")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return
    
    try:
        # Read the image
        frame = cv2.imread(image_path)
        
        if frame is None:
            print("ERROR: Cannot read image")
            return
        
        print("Running detection...")
        
        # Run inference
        results = model(frame, conf=confidence)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow("SignSense AI - Image Detection", annotated_frame)
        
        # Print detections
        print("\nDetections:")
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = results[0].names[cls]
                print(f"  ✓ {label} (Confidence: {conf:.2f})")
        else:
            print("  No signs detected")
        
        # Save the annotated image
        output_path = "detection_result.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"\nAnnotated image saved to {output_path}")
        
        # Wait for keypress to close
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"ERROR: {e}")


def detect_from_folder(model, folder_path="images", confidence=0.5):
    """
    Run detection on all images in a folder
    
    Args:
        model: Loaded YOLOv5 model
        folder_path: Path to folder containing images
        confidence: Confidence threshold (0-1)
    """
    if model is None:
        return
    
    print(f"Processing images from {folder_path}...")
    
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder not found at {folder_path}")
        return
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in os.listdir(folder_path) 
                   if Path(f).suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} image(s)\n")
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"\nProcessing: {image_file}")
        detect_from_image(model, image_path, confidence)


def main():
    """
    Main function to run SignSense AI
    """
    print("=" * 60)
    print("🧠 SignSense AI - Sign Language Detection")
    print("=" * 60)
    print()
    
    # Load the model
    model = load_model("model/best.pt")
    
    if model is None:
        print("\n⚠️  Please ensure your trained model is placed at: model/best.pt")
        return
    
    print("\nChoose detection mode:")
    print("1. Real-time Webcam Detection (Main)")
    print("2. Detect from single Image")
    print("3. Detect from Folder (images/)")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        detect_from_webcam(model)
    elif choice == "2":
        image_path = input("Enter image path: ").strip()
        detect_from_image(model, image_path)
    elif choice == "3":
        detect_from_folder(model)
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()

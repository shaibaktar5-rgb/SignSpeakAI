# 🧠 SignSense AI - Real-time Sign Language Detection

A simple Python-based system for detecting sign language gestures using YOLOv5 and OpenCV.

## 📋 Project Structure

```
SignSpeakAI/
├── model/
│   └── best.pt          # Your trained YOLOv5 model (add this)
├── images/              # Test images folder
├── detect.py            # Main detection script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### 1. **Set Up Environment**

```bash
# Navigate to project folder
cd SignSpeakAI

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Add Your Model**

Place your trained YOLOv5 model file (`best.pt`) in the `model/` folder:
```
SignSpeakAI/model/best.pt
```

### 3. **Run Detection**

```bash
python detect.py
```

Choose from 3 options:
- **Option 1**: Real-time Webcam Detection (main method)
- **Option 2**: Single Image Detection
- **Option 3**: Batch Detection from `images/` folder

## 🎮 Usage Modes

### Mode 1: Real-Time Webcam Detection
```bash
python detect.py
# Select option 1
# Webcam opens → Shows live sign detection
# Press 'q' to quit
```

**What happens:**
- Camera captures frames in real-time
- Each frame is sent to YOLO model
- Detected signs appear with bounding boxes
- Confidence scores shown on screen

### Mode 2: Single Image Detection
```bash
python detect.py
# Select option 2
# Enter image path (e.g., images/test.jpg)
# Shows detection results
# Saves annotated image as detection_result.jpg
```

### Mode 3: Batch Processing
```bash
python detect.py
# Select option 3
# Processes all images in images/ folder
# Shows results for each image
```

## 🔧 How It Works

```
Camera/Image Input
       ↓
   OpenCV reads frame
       ↓
  YOLOv5 model runs inference
       ↓
  Model returns detections:
  - Bounding box coordinates
  - Predicted sign label
  - Confidence score
       ↓
  Draw boxes & labels on frame
       ↓
  Display output
       ↓
  (Repeat for next frame)
```

## ⚙️ Configuration

### Adjust Confidence Threshold

Edit `detect.py` and modify the `confidence` parameter:

```python
detect_from_webcam(model, confidence=0.5)  # 0-1 range
```

- **0.5** = More detections, some false positives
- **0.7** = Balanced
- **0.9** = Only confident detections

### Camera Selection

If you have multiple cameras, modify line in `detect.py`:

```python
cap = cv2.VideoCapture(0)  # 0 = default, 1 = second camera, etc.
```

## 📝 Example: Adding Test Images

1. Add your test images to `images/` folder:
```
SignSpeakAI/images/
├── sign1.jpg
├── sign2.jpg
└── sign3.jpg
```

2. Run batch detection:
```bash
python detect.py
# Select option 3
```

## 🐛 Troubleshooting

### Issue: "Model not found at model/best.pt"
**Solution:** Make sure your trained model is placed in the `model/` folder

### Issue: "Cannot open webcam"
**Solution:**
- Check if camera is not being used by another app
- Try switching camera: `cv2.VideoCapture(1)`
- On Mac/Linux, may need camera permissions

### Issue: Slow performance
**Solution:**
- Lower frame resolution
- Reduce model inference size
- Use GPU if available (install CUDA)

### Issue: No detections
**Solution:**
- Check confidence threshold (lower it)
- Ensure model was trained on similar hand signs
- Make sure lighting is good

## 📚 Dependencies Explained

| Package | Purpose |
|---------|---------|
| `opencv-python` | Read webcam & images |
| `torch` | Deep learning framework |
| `torchvision` | Vision utilities |
| `ultralytics` | YOLOv5 implementation |
| `numpy` | Numerical computing |
| `Pillow` | Image processing |

## 🎯 Success Checklist

- [x] Project structure created
- [x] Dependencies listed in requirements.txt
- [x] detect.py works with 3 detection modes
- [ ] Place your `best.pt` model in `model/` folder
- [ ] Test with webcam
- [ ] Test with sample images

## 💡 Next Steps (Optional)

1. **Add voice feedback**: Text-to-speech for detected signs
2. **Sign to text**: Convert detected signs → words/sentences
3. **Web interface**: Create Flask/Streamlit web app
4. **Performance**: Optimize for speed, add FPS counter
5. **Mobile**: Deploy to mobile using TensorFlow Lite

## 🔗 Useful Resources

- [YOLOv5 Documentation](https://github.com/ultralytics/yolov5)
- [OpenCV Tutorials](https://docs.opencv.org/)
- [PyTorch Getting Started](https://pytorch.org/get-started/locally/)

## ✨ Summary

Your sign language detection system is ready! Just:
1. ✅ Install requirements: `pip install -r requirements.txt`
2. ✅ Add your trained model: `model/best.pt`
3. ✅ Run: `python detect.py`
4. ✅ Choose detection mode (1, 2, or 3)
5. ✅ Watch your AI detect signs! 🚀

---

**Happy Detecting! 🧠✨**

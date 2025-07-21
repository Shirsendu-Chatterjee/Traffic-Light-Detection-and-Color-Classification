# Traffic-Light-Detection-and-Color-Classification
This project uses the **YOLOv8 object detection model** (Ultralytics) to detect traffic lights in videos and classify their color (Red, Yellow, Green) using **HSV color segmentation** with OpenCV. Based on the detected light color, the system provides an action label: **STOP**, **SLOW**, or **GO**.

## 🎯 Objective

To build a real-time traffic light detection system that can:
- Detect traffic lights in video input
- Classify light color using HSV filtering
- Display the appropriate action command for autonomous driving simulations

---

## 🧠 Model & Approach

- **Object Detection:** YOLOv8n (`yolov8n.pt`) pretrained weights
- **Color Classification:** HSV masking in OpenCV
- **Stabilization:** Uses most frequent color in a sliding window to reduce flicker
- **Output Actions:** STOP (Red), SLOW (Yellow), GO (Green)

---

## 📂 File Structure

```bash
📦 traffic-light-detection/
 ┣ 📜 detect_traffic_light.py      # Main script
 ┣ 📂 video_input/                 # Input videos (e.g., WhatsApp traffic video)
 ┣ 📂 ultralytics/                 # YOLOv8 model code (via pip)
 ┣ 📜 requirements.txt             # Dependencies

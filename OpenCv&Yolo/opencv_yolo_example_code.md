# OpenCV + YOLO Example Code: Real-Time Object Detection

## 🔍 Core Concept

YOLO (You Only Look Once) is a real-time object detection algorithm that processes images and detects multiple objects in one pass. OpenCV provides a flexible way to load pre-trained YOLO models and perform inference on live video or images.

## 🧠 Key Concepts

- YOLO splits images into grids and predicts bounding boxes and class probabilities.
- OpenCV's DNN module allows loading `.cfg`, `.weights`, and `.names` files to run YOLO inference.
- Real-time detection can be achieved with webcam or video file input.

## 🚀 Sample Code

```python
import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
img = cv2.imread("sample.jpg")
height, width = img.shape[:2]

# Prepare input
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Draw boxes
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Show result
cv2.imshow("YOLO Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 📌 Prompt Examples

- “How do I detect objects using YOLO and OpenCV?”
- “What are the input files required to run YOLOv3?”

## 🖼️ Related Image/Video Ideas

- YOLO grid detection heatmap.
- Screenshot of bounding boxes labeled on live webcam feed.

## 🗣️ Audio Prompt Hook

- “How does YOLO detect objects in real-time?”
- “What are the steps to run object detection on a video?”

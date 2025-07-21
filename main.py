from ultralytics import YOLO
import cv2
import numpy as np

model=YOLO("yolov8n.pt")  

COLOR_RANGES = {
    'Red':    [(0, 100, 100), (10, 255, 255)],
    'Yellow': [(20, 100, 100), (30, 255, 255)],
    'Green':  [(40, 50, 50), (90, 255, 255)]
}

def detect_color(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_detected = "undefined"    
    max=0

    for color, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask= cv2.inRange(hsv_roi, lower, upper)
        color_area = cv2.countNonZero(mask)
        if color_area > max:
            max=color_area
            color_detected = color
    
    return color_detected

from collections import Counter

def most_common_string(strings):
    count = Counter(strings)
    most_common = count.most_common(1)
    return most_common[0][0] if most_common else strings[8]



cap = cv2.VideoCapture('WhatsApp Video 2025-05-03 at 11.25.48 PM.mp4')
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

c=['undefined','undefined','undefined','undefined','undefined','undefined','undefined','undefined','undefined']
prev='undefined'

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame,conf=0.15)[0]
    for box in results.boxes:
        if model.names[int(box.cls[0])] == "traffic light":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]
            color_detected = detect_color(roi)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (4, 55, 255), 1)
            c.append(color_detected)
            c.pop(0)
            mcc = most_common_string(c)
            cv2.putText(frame, mcc, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (4, 55, 255), 1)
            if mcc== 'undefined':
                mcc=prev
            if mcc == 'Red':
                cv2.putText(frame, "STOP", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (4, 55, 255), 2)
                prev='Red'
            elif mcc == 'Yellow':
                cv2.putText(frame, "SLOW", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (4, 255, 255), 2)
                prev='Yellow'
            elif mcc == 'Green':
                cv2.putText(frame, "GO", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (4, 255, 55), 2)
                prev='Green'
            else:
                cv2.putText(frame, "undefined", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (4, 255, 255), 2)

        
    cv2.imshow("Traffic Light Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

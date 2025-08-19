import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

yolo_v8 = YOLO("yolov8n.pt")

# video3_path = '../single_cv/videos/walk_dog.mp4'
video3_path = "C:/Users/Tarek/Desktop/Computer Vision/Object Tracking/single_cv/videos/walk_dog.mp4"

video = cv2.VideoCapture(video3_path)
height = 400
width = 600

while True:
    # get the current frame
    check, frame = video.read()
    if not check:
        print(f"Failed to grab the frame, please check you video!!")
        break
    # resized the frame with expected size 
    resized_frame = cv2.resize(frame, (width, height))
    pred_results = yolo_v8(resized_frame)
    
    for result in pred_results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            cls = box.cls[0].cpu().numpy()  # Class index
            label = result.names[int(cls)]  # Get class label
            if label == "person":
                conf = box.conf[0].cpu().numpy()  # Confidence score
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get coordinates (x1, y1, x2, y2)
                
                # Draw bounding box on the frame
                color = (0, 255, 0)  # Green color for boxes (you can change this)
                cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
                # Put the label (class name) and confidence score on the frame
                text = f"{label} {conf:.2f}"
                cv2.putText(resized_frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # showing the video
    cv2.imshow('video', resized_frame)
    
    key = cv2.waitKey(1)
    # press Esc to break 
    if key == 27:
        break
    pass

video.release()
cv2.destroyAllWindows()